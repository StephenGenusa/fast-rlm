"""
Core RLM engine.

`RLM.complete(query, context)` is a drop-in replacement for a single LLM call:
same input space (a query + some context), same output space (a string / value),
but the context can be arbitrarily large because it is stored as a REPL variable
the model explores with code instead of being stuffed into the prompt.

Every model interaction is recorded as a tracing Span (see rlm/tracing.py); the
full trace is returned on `RLMResult.trace` and streamed via `RLM(on_span=...)`.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .clients import Completion, LLMClient, Usage
from .prompts import DEFAULT_QUERY, build_messages, step_prompt
from .tracing import Span, Tracer


def extract_repl_blocks(text: str) -> list[str]:
    # Prefer the canonical ```repl fence. Only if a turn has none, fall back to
    # ```python / ```py so a model that forgets the tag still makes progress —
    # but never run ```python in a turn that already has a ```repl block (avoids
    # executing illustrative code shown alongside a real action).
    repl = [m.group(1).strip()
            for m in re.finditer(r"```repl\s*\n(.*?)```", text, re.DOTALL)]
    if repl:
        return repl
    return [m.group(1).strip()
            for m in re.finditer(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL)]


def extract_text_final(text: str) -> tuple[bool, str | None]:
    for kw in ("FINAL_VAR", "FINAL"):
        idx = 0
        while True:
            i = text.find(kw + "(", idx)
            if i == -1:
                break
            if kw == "FINAL" and text[max(0, i - 4):i].endswith("_VAR"):
                idx = i + 1
                continue
            j = i + len(kw) + 1
            depth, in_str, quote = 1, False, ""
            while j < len(text) and depth:
                c = text[j]
                if in_str:
                    if c == quote and text[j - 1] != "\\":
                        in_str = False
                elif c in "\"'":
                    in_str, quote = True, c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                return True, text[i + len(kw) + 1: j - 1].strip()
            idx = i + 1
    return False, None


@dataclass
class RLMResult:
    answer: Any
    usage: Usage
    iterations: int
    trace: list[dict] = field(default_factory=list)
    trace_id: str | None = None


class RLM:
    """Recursive Language Model: a thin, recursive wrapper around an LLM."""

    def __init__(self, client: LLMClient, sub_client: LLMClient | None = None,
                 sandbox=None, max_iterations: int = 12, max_depth: int = 1,
                 sub_max_parallel: int = 8, verbose: bool = False,
                 tools: dict[str, Callable] | None = None, sandbox_pool=None,
                 on_span: Callable[[Span], None] | None = None,
                 trace_content: bool = False,
                 redact: Callable[[str], str] | None = None,
                 trace_dir: str | None = "rlm_traces"):
        self.client = client
        self.sub_client = sub_client or client
        self.sandbox = sandbox  # default constructed lazily in complete()
        self.sandbox_pool = sandbox_pool  # if set, lease per call instead of own
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.sub_max_parallel = sub_max_parallel
        self.verbose = verbose
        self.tools = dict(tools or {})
        self.on_span = on_span            # live span callback (TUI/logging)
        self.trace_content = trace_content  # capture prompt/response text in spans (off by default)
        self.redact = redact              # optional scrubber for captured text
        self.trace_dir = trace_dir        # autosave dir (used only when trace_content)
        self.usage = Usage()
        self._usage_lock = threading.Lock()  # batch sub-calls add usage from threads

    def _add_usage(self, u: Usage) -> None:
        with self._usage_lock:
            self.usage.add(u)

    # -- schema validation ---------------------------------------------------
    def _finalize_schema(self, text: Any, schema: dict | None,
                         tracer: Tracer | None = None, parent_id: str | None = None) -> Any:
        if schema is None:
            return text
        obj = _coerce(text, schema)
        ok, err = _validate(obj, schema)
        if ok:
            return obj
        repair = (
            "Your previous output did not satisfy the required JSON schema.\n"
            f"JSON schema:\n{json.dumps(schema)}\n"
            f"Validation error: {err}\n"
            f"Previous output:\n{text}\n\n"
            "Return ONLY corrected JSON that satisfies the schema, with no prose.")
        if tracer is not None:
            with tracer.span("schema_repair", "schema repair", parent_id=parent_id,
                             model=getattr(self.sub_client, "model", None)) as sp:
                comp = self.sub_client.complete([{"role": "user", "content": repair}])
                self._add_usage(comp.usage)
                sp.input_tokens, sp.output_tokens = comp.usage.input_tokens, comp.usage.output_tokens
                sp.cost_usd, sp.latency_s = comp.usage.cost_usd, comp.latency_s
                sp.content = tracer.content(response=comp.text)
        else:
            comp = self.sub_client.complete([{"role": "user", "content": repair}])
            self._add_usage(comp.usage)
        obj2 = _coerce(comp.text, schema)
        ok2, _ = _validate(obj2, schema)
        return obj2 if ok2 else obj

    # -- host-mediated tools -------------------------------------------------
    def _call_tool(self, name: str, args, kwargs) -> Any:
        fn = self.tools.get(name)
        if fn is None:
            return f"[tool error: unknown tool {name!r}; available: {sorted(self.tools)}]"
        try:
            result = fn(*list(args), **dict(kwargs))
        except Exception as e:  # surface tool failures into the REPL  # noqa
            return f"[tool error: {type(e).__name__}: {e}]"
        try:
            json.dumps(result)
            return result
        except (TypeError, ValueError):
            return str(result)

    # -- the channel the sandbox calls back into for sub-LM work -------------
    def _make_sub_handler(self, depth: int, active_sandbox=None,
                          tracer: Tracer | None = None,
                          parent_box: list | None = None) -> Callable[[str, Any], Any]:
        tracer = tracer or Tracer()            # detached tracer if called bare (tests)
        parent_box = parent_box if parent_box is not None else [None]

        def _leaf(task: str, data: Any, schema: dict | None,
                  kind: str = "sub_query") -> Any:
            content = task if data is None else f"{task}\n\n--- DATA ---\n{_as_text(data)}"
            name = {"sub_query": "llm_query", "sub_batch_item": "batch_item"}.get(kind, kind)
            with tracer.span(kind, name, parent_id=parent_box[0], depth=depth + 1,
                             model=getattr(self.sub_client, "model", None)) as sp:
                comp = self.sub_client.complete([{"role": "user", "content": content}])
                self._add_usage(comp.usage)
                sp.input_tokens, sp.output_tokens = comp.usage.input_tokens, comp.usage.output_tokens
                sp.cost_usd, sp.latency_s = comp.usage.cost_usd, comp.latency_s
                sp.content = tracer.content(prompt=content, response=comp.text)
                result_text = comp.text
            return self._finalize_schema(result_text, schema, tracer, parent_box[0])

        def _recursive(task: str, data: Any, schema: dict | None,
                       kind: str = "sub_query") -> Any:
            if depth + 1 >= self.max_depth:
                return _leaf(task, data, schema, kind)   # at the budget: fall back to a leaf
            # recursive child gets its OWN fresh sandbox (isolated context, globals,
            # pipe); its spans nest into the shared tracer under the current cell.
            child_sb = active_sandbox.clone() if active_sandbox is not None else None
            child = RLM(self.sub_client, self.sub_client, sandbox=child_sb,
                        max_iterations=self.max_iterations, max_depth=self.max_depth,
                        sub_max_parallel=self.sub_max_parallel, tools=self.tools,
                        on_span=self.on_span, trace_content=self.trace_content,
                        redact=self.redact)
            try:
                res = child.complete(task, data if data is not None else "",
                                     _depth=depth + 1, _tracer=tracer,
                                     _parent_id=parent_box[0])
            finally:
                if child_sb is not None:
                    child_sb.close()
            self._add_usage(res.usage)
            return self._finalize_schema(res.answer, schema, tracer, parent_box[0])

        def _map(fn, payload, item_kind):
            jobs = list(payload)
            with ThreadPoolExecutor(max_workers=self.sub_max_parallel) as ex:
                return list(ex.map(
                    lambda j: fn(j.get("task", ""), j.get("data"), j.get("schema"), item_kind),
                    jobs))

        def handler(kind: str, payload: Any) -> Any:
            if kind == "query":
                return _leaf(payload.get("task", ""), payload.get("data"), payload.get("schema"))
            if kind == "batch":
                return _map(_leaf, payload, "sub_batch_item")
            if kind == "rlm_query":
                return _recursive(payload.get("task", ""), payload.get("data"),
                                  payload.get("schema"))
            if kind == "rlm_batch":
                return _map(_recursive, payload, "sub_query")
            if kind == "tool":
                name = payload.get("name", "")
                with tracer.span("tool", f"tool:{name}", parent_id=parent_box[0],
                                 depth=depth + 1) as sp:
                    out = self._call_tool(name, payload.get("args") or [],
                                          payload.get("kwargs") or {})
                    sp.content = tracer.content(args=str(payload.get("args")),
                                                kwargs=str(payload.get("kwargs")), result=out)
                    if isinstance(out, str) and out.startswith("[tool error"):
                        sp.error = out
                return out
            raise ValueError(f"unknown sub-call kind {kind!r}")

        return handler

    def complete(self, query: str | None, context: Any, _depth: int = 0,
                 _tracer: Tracer | None = None, _parent_id: str | None = None) -> RLMResult:
        query = query or DEFAULT_QUERY
        start = Usage(); start.add(self.usage)

        tracer = _tracer or Tracer(on_span=self.on_span, trace_content=self.trace_content,
                                   redact=self.redact)
        root = tracer.open("root" if _depth == 0 else "rlm", f"rlm(depth={_depth})",
                           parent_id=_parent_id, depth=_depth,
                           model=getattr(self.client, "model", None))
        parent_box = [root.span_id]

        pool = self.sandbox_pool
        if pool is not None:
            sandbox = pool.acquire()
            owns_sandbox = False
            from_pool = True
        else:
            sandbox = self.sandbox
            owns_sandbox = sandbox is None
            from_pool = False
            if owns_sandbox:
                from .sandbox import PyodideSandbox  # default to the real sandbox
                sandbox = PyodideSandbox()
        sandbox.start()  # idempotent: starts if a caller passed an unstarted sandbox
        sandbox.set_context(context)

        sub_handler = self._make_sub_handler(_depth, sandbox, tracer, parent_box)
        messages = build_messages(query, _context_meta(context),
                                  async_subcalls=getattr(sandbox, "async_subcalls", True),
                                  tools_doc=_tools_doc(self.tools),
                                  recursive_enabled=self.max_depth > 1)
        last_iter = 0
        answer = None
        try:
            for it in range(self.max_iterations):
                last_iter = it
                step = step_prompt(query, it)
                with tracer.span("root_turn", f"turn {it}", parent_id=root.span_id,
                                 depth=_depth, model=getattr(self.client, "model", None)) as sp:
                    comp = self.client.complete(messages + [step])
                    self._add_usage(comp.usage)
                    sp.input_tokens, sp.output_tokens = comp.usage.input_tokens, comp.usage.output_tokens
                    sp.cost_usd, sp.latency_s = comp.usage.cost_usd, comp.latency_s
                    sp.content = tracer.content(prompt=step.get("content"), response=comp.text)
                text = comp.text
                if self.verbose:
                    print(f"\n=== root iter {it} (depth {_depth}) ===\n{text[:1500]}")

                blocks = extract_repl_blocks(text)
                messages.append({"role": "assistant", "content": text})

                done = False
                for code in blocks:
                    with tracer.span("repl_cell", f"cell {it}", parent_id=root.span_id,
                                     depth=_depth) as sp:
                        parent_box[0] = sp.span_id  # sub-calls in this cell nest here
                        result = sandbox.run(code, sub_handler)
                        sp.content = tracer.content(code=code, stdout=result.stdout)
                    parent_box[0] = root.span_id
                    messages.append({"role": "user",
                                     "content": f"REPL output:\n{result.stdout}"})
                    if result.has_final:
                        answer, done = result.final, True
                        break
                if done:
                    break

                found, expr = extract_text_final(text)
                if found and not blocks:
                    answer, done = expr, True
                    break
                if not blocks:
                    messages.append({"role": "user",
                                     "content": "No code block found. Write a ```repl``` "
                                                "block (```python/```py also accepted) or "
                                                "call FINAL(...)."})
            else:
                step = step_prompt(query, last_iter, force_final=True)
                with tracer.span("root_turn", "forced final", parent_id=root.span_id,
                                 depth=_depth, model=getattr(self.client, "model", None)) as sp:
                    comp = self.client.complete(messages + [step])
                    self._add_usage(comp.usage)
                    sp.input_tokens, sp.output_tokens = comp.usage.input_tokens, comp.usage.output_tokens
                    sp.cost_usd, sp.latency_s = comp.usage.cost_usd, comp.latency_s
                    sp.content = tracer.content(prompt=step.get("content"), response=comp.text)
                found, expr = extract_text_final(comp.text)
                answer = expr if found else comp.text
        finally:
            if from_pool:
                pool.release(sandbox)
            elif owns_sandbox:
                sandbox.close()

        run_usage = Usage(self.usage.input_tokens - start.input_tokens,
                          self.usage.output_tokens - start.output_tokens,
                          self.usage.calls - start.calls,
                          self.usage.cost_usd - start.cost_usd)
        root.input_tokens, root.output_tokens = run_usage.input_tokens, run_usage.output_tokens
        root.cost_usd = run_usage.cost_usd
        tracer.close(root)
        if _tracer is None and self.trace_content and self.trace_dir:
            # autosave the completed trace (best-effort; never fail the run)
            try:
                import os
                tracer.save(os.path.join(self.trace_dir, f"{tracer.trace_id}.jsonl"))
            except Exception as e:  # noqa
                import sys
                sys.stderr.write(f"[rlm] WARNING: failed to autosave trace: {e}\n")
        return RLMResult(answer, run_usage, last_iter + 1, tracer.to_list(), tracer.trace_id)


# --- helpers -----------------------------------------------------------------
def _tools_doc(tools: dict) -> str:
    if not tools:
        return ""
    lines = []
    for name, fn in tools.items():
        doc = ((getattr(fn, "__doc__", "") or "").strip().split("\n") or [""])[0]
        lines.append(f"- {name}: {doc}" if doc else f"- {name}")
    return "\n".join(lines)


def _context_meta(context: Any) -> str:
    if isinstance(context, str):
        head = context[:300].replace("\n", " ")
        return (f"type=str, length={len(context):,} chars. "
                f"First 300 chars: {head!r}")
    if isinstance(context, dict):
        return f"type=dict, top-level keys={list(context)[:20]}"
    if isinstance(context, list):
        return f"type=list, length={len(context):,} items"
    return f"type={type(context).__name__}"


def _as_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, ensure_ascii=False)[:500_000]
    except TypeError:
        return str(data)[:500_000]


def _coerce(text: Any, schema: dict | None) -> Any:
    if schema is None or not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except ValueError:
                pass
    return text


def _validate(obj: Any, schema: dict | None) -> tuple[bool, str | None]:
    """Validate against a JSON schema if `jsonschema` is installed; else pass."""
    if schema is None:
        return True, None
    try:
        import jsonschema
    except ImportError:
        return True, None  # parsing-only fallback; cannot enforce
    try:
        jsonschema.validate(obj, schema)
        return True, None
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        first = str(e).splitlines()[0] if str(e) else "validation failed"
        return False, first
    except Exception as e:  # bad schema, etc.  # noqa
        return False, f"{type(e).__name__}: {e}"
