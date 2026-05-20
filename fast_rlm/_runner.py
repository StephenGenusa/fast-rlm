import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


_PRIMITIVE_JSON_SCHEMAS = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


def _to_json_schema(schema: Any) -> dict:
    """Convert a user-supplied output schema into a JSON Schema dict.

    Accepts:
      - dict: assumed to already be a JSON Schema, returned as-is.
      - Python primitive type (str/int/float/bool/list/dict): mapped directly.
      - Pydantic BaseModel subclass: uses model_json_schema().
      - Any other type usable by pydantic.TypeAdapter (e.g. list[int]): uses TypeAdapter.
    """
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type) and schema in _PRIMITIVE_JSON_SCHEMAS:
        return _PRIMITIVE_JSON_SCHEMAS[schema]
    try:
        from pydantic import BaseModel, TypeAdapter
    except ImportError as e:
        raise TypeError(
            "output_schema must be a JSON Schema dict, a primitive type "
            "(str/int/float/bool/list/dict), or a pydantic type. "
            "Install pydantic to use Pydantic models or generic types."
        ) from e
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    return TypeAdapter(schema).json_schema()


@dataclass
class RLMConfig:
    """Configuration for fast-rlm."""

    primary_agent: str = "z-ai/glm-5"
    sub_agent: str = "minimax/minimax-m2.5"
    max_depth: int = 3
    max_calls_per_subagent: int = 20
    truncate_len: int = 2000
    max_money_spent: float = 1.0
    max_completion_tokens: int = 50000
    max_prompt_tokens: int = 200000
    api_max_retries: int = 3
    api_timeout_ms: int = 600000

    @classmethod
    def default(cls) -> "RLMConfig":
        """Load defaults from bundled rlm_config.yaml."""
        try:
            engine_dir = _find_engine_dir()
            config_path = engine_dir / "rlm_config.yaml"
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            return cls(
                **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            )
        except Exception:
            return cls()


def _find_engine_dir() -> Path:
    """Find the TS engine: bundled (_engine/) for pip install, project root for dev."""
    bundled = Path(__file__).parent / "_engine"
    if (bundled / "deno.json").exists():
        return bundled

    # Editable / dev install: walk up to project root
    root = Path(__file__).resolve().parent.parent
    if (root / "deno.json").exists():
        return root

    raise FileNotFoundError(
        "Cannot find the fast-rlm TS engine. "
        "Ensure the package is installed correctly or you're in the project root."
    )


def _check_deno():
    if shutil.which("deno") is None:
        raise RuntimeError(
            "Deno is required but not found on PATH.\n"
            "Install it on (Mac/Linux): curl -fsSL https://deno.land/install.sh | sh\n"
            "Install it on (Windows): npm install -g deno\n"
            "Visit: https://docs.deno.com/runtime/getting_started/installation/ for more installation options"
        )

def _deno_prefix_cmd() -> list[str]:
    """Return a subprocess-safe Deno command prefix across platforms."""
    deno = shutil.which("deno")
    if deno is None:
        raise RuntimeError("Deno is required but not found on PATH.")

    # npm installs on Windows often expose deno as a .cmd shim.
    # Python's subprocess can fail on that directly, so route via cmd.exe.
    if os.name == "nt" and deno.lower().endswith(".cmd"):
        return ["cmd", "/c", deno]
    return [deno]


def run(
    query: "str | dict",
    prefix: Optional[str] = None,
    config: Optional[RLMConfig | dict] = None,
    verbose: bool = True,
    output_schema: Optional[Any] = None,
) -> dict:
    """Run a fast-rlm query.

    Args:
        query: The question / context to process. Either a string or a JSON-
            serializable dict — when a dict, the agent receives `context` as a
            real Python dict and the initial probe prints its top-level schema.
        prefix: Optional log filename prefix.
        config: RLMConfig object or dict of overrides (e.g. primary_agent, max_depth).
        verbose: If True, stream deno stdout/stderr to terminal.
        output_schema: Optional schema the root agent's FINAL value must satisfy.
            Accepts a Pydantic model class, a primitive Python type (str/int/
            float/bool/list/dict), a `pydantic.TypeAdapter`-compatible type, or
            a raw JSON Schema dict. Validation runs after FINAL is set; on
            failure the agent receives the schema + errors and may retry within
            its remaining call budget.

    Returns:
        Dict with 'results', 'usage', and optionally 'log_file'.
    """
    _check_deno()
    engine_dir = _find_engine_dir()

    output_file = tempfile.mktemp(suffix=".json")
    log_dir = os.path.join(os.getcwd(), "logs")

    cmd = _deno_prefix_cmd() + [
        "run",
        "--allow-read",
        "--allow-env",
        "--allow-net",
        "--allow-sys=hostname,osRelease",
        "--allow-write",
        "src/subagents.ts",
        "--log-dir",
        log_dir,
        "--output",
        output_file,
        "--input-json",
    ]

    if prefix:
        cmd += ["--prefix", prefix]

    if not isinstance(query, (str, dict)):
        raise TypeError(
            f"query must be a str or dict, got {type(query).__name__}"
        )
    stdin_payload = json.dumps(query)

    schema_tmpfile = None
    if output_schema is not None:
        schema_dict = _to_json_schema(output_schema)
        schema_tmpfile = tempfile.mktemp(suffix=".schema.json")
        with open(schema_tmpfile, "w") as f:
            json.dump(schema_dict, f)
        cmd += ["--output-schema-file", schema_tmpfile]

    # RLMConfig merge: load defaults, overlay user overrides, write to temp file
    config_tmpfile = None
    if config is not None:
        if isinstance(config, RLMConfig):
            config = asdict(config)

        default_config_path = engine_dir / "rlm_config.yaml"
        defaults = {}
        if default_config_path.exists():
            with open(default_config_path) as f:
                defaults = yaml.safe_load(f) or {}
        merged = {**defaults, **config}

        config_tmpfile = tempfile.mktemp(suffix=".yaml")
        with open(config_tmpfile, "w") as f:
            yaml.dump(merged, f)
        cmd += ["--config", config_tmpfile]

    try:
        result = subprocess.run(
            cmd,
            input=stdin_payload,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(engine_dir),
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
        )

        if not os.path.exists(output_file):
            stderr = result.stderr or "" if not verbose else ""
            raise RuntimeError(
                f"fast-rlm engine failed (exit code {result.returncode}).\n{stderr}"
            )

        with open(output_file) as f:
            data = json.load(f)
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)
        if config_tmpfile and os.path.exists(config_tmpfile):
            os.unlink(config_tmpfile)
        if schema_tmpfile and os.path.exists(schema_tmpfile):
            os.unlink(schema_tmpfile)

    if "error" in data:
        raise RuntimeError(f"fast-rlm subagent failed: {data['error']}")

    return data
