# fast-rlm

[![PyPI](https://img.shields.io/pypi/v/fast-rlm)](https://pypi.org/project/fast-rlm/)
[![GitHub](https://img.shields.io/github/stars/avbiswas/fast-rlm)](https://github.com/avbiswas/fast-rlm)
[![Docs](https://img.shields.io/badge/docs-avbiswas.github.io%2Ffast--rlm-blue)](https://avbiswas.github.io/fast-rlm/)

A minimal implementation of Recursive Language Models (RLMs) using Deno and Pyodide.

[GitHub](https://github.com/avbiswas/fast-rlm) | [Documentation](https://avbiswas.github.io/fast-rlm/) | [PyPI](https://pypi.org/project/fast-rlm/)

> **Watch the full video on YouTube**
> **[RLM Tutorial](https://youtu.be/nxaVvvrezbY)**

## What are RLMs

RLMs are an inference technique where an LLM interacts with arbitrarily long prompts through an external REPL. The LLM can write code to explore, decompose, and transform the prompt. It can recursively invoke sub-agents to complete smaller subtasks. Crucially, sub-agent responses are not automatically loaded into the parent agent's context — they are returned as symbols or variables inside the parent's REPL.

## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

---

## Install

```bash
pip install fast-rlm
```

### Requirements

- Python 3.10+
- [Deno](https://deno.land/) 2+
  - macOS/Linux: `curl -fsSL https://deno.land/install.sh | sh`
  - Windows (npm): `npm install -g deno`
- (Optional) [Bun](https://bun.sh/) — only needed for the TUI log viewer

### Environment Variables

Set your LLM API key before running:

```bash
export RLM_MODEL_API_KEY=sk-or-...
```

| Variable | Description | Default |
|----------|-------------|---------|
| `RLM_MODEL_API_KEY` | API key for your LLM provider | — |
| `RLM_MODEL_BASE_URL` | OpenAI-compatible base URL | `https://openrouter.ai/api/v1` |

By default, fast-rlm uses [OpenRouter](https://openrouter.ai). You can point it at any OpenAI-compatible API by setting `RLM_MODEL_BASE_URL`.

---

## Quick Start

![Quickstart](docs/images/quickstart.jpeg)

```python
import fast_rlm

result = fast_rlm.run("Generate 50 fruits and count number of r")
print(result["results"])
print(result["usage"])
```

## Arbitrarily Long Context

The key idea behind RLMs is that the prompt can be arbitrarily long — far beyond any model's context window. The agent explores it programmatically through the REPL rather than trying to fit it all into a single call.

```python
import fast_rlm

transcripts = open("lex_fridman_all_transcripts.txt").read()  # millions of tokens

result = fast_rlm.run(
    "Here are the transcripts of all Lex Fridman podcasts. "
    "Summarize what the first 5 Machine Learning guests had to say about AGI.\n\n"
    + transcripts
)
print(result["results"])
```

The agent will write code to search, filter, and chunk the transcripts on its own — no manual splitting required.

## Structured Input & Output

Instead of squeezing your data into a string, you can pass a `dict` as the query and ask for a typed result back via `output_schema`. The agent receives the dict as a real Python `dict` (no parsing on its first turn), and its `FINAL` value is validated against the schema before being returned.

```python
import fast_rlm
from pydantic import BaseModel

class Verdict(BaseModel):
    movie: str
    average_score: float
    consensus: str

result = fast_rlm.run(
    {
        "task": "Aggregate the reviews into a single verdict.",
        "movie": "The Trail of Pixels",
        "reviews": [
            {"name": "Asha", "score": 8, "text": "Tight pacing..."},
            {"name": "Bo",   "score": 6, "text": "Beautiful but thin..."},
            {"name": "Cy",   "score": 9, "text": "Instant favorite..."},
        ],
    },
    output_schema=Verdict,
)

verdict = Verdict.model_validate(result["results"])
```

**Structured input.** When `query` is a `dict`, the agent's initial probe prints a flat top-level schema (keys + type + length + truncated preview) so it can index `context["reviews"]` directly instead of stringifying.

**Structured output.** `output_schema` accepts:

| Form | Example |
|---|---|
| Pydantic model class | `output_schema=MyModel` |
| Pydantic generic | `output_schema=list[MyModel]` |
| Python primitive | `output_schema=int` (also `str`, `float`, `bool`, `list`, `dict`) |
| Raw JSON Schema dict | `output_schema={"type": "array", "items": {"type": "string"}}` |

The schema is shown to the agent at step 0 (`Required output schema for FINAL (JSON Schema):`). After every `FINAL(...)` call the value is validated; on failure the agent receives the schema and the specific validation errors (path + message) and may retry within its remaining call budget. Pydantic is an *optional* dependency — only required if you pass a Pydantic class or generic.

**Schemas for subagents.** Inside the REPL the agent can require a subagent's output shape by passing a JSON Schema dict as the second argument to `llm_query`:

```repl
schema = {"type": "array", "items": {"type": "string"}}
fruits = await llm_query("Generate 25 fruit names.", schema)
```

The child subagent enforces the schema the same way. See [`examples/structured_io.py`](examples/structured_io.py) and [`examples/parallel_r_count.py`](examples/parallel_r_count.py) for end-to-end demos.

## Configuration

```python
from fast_rlm import run, RLMConfig

config = RLMConfig.default()
config.primary_agent = "minimax/minimax-m2.5"
config.sub_agent = "minimax/minimax-m2.5"
config.max_depth = 5
config.max_money_spent = 2.0

result = run(
    "Count the r's in 50 fruit names",
    prefix="r_count",
    config=config,
)
```

All config fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `primary_agent` | `str` | `z-ai/glm-5` | Model for the root agent |
| `sub_agent` | `str` | `minimax/minimax-m2.5` | Model for child subagents |
| `max_depth` | `int` | `3` | Max recursive subagent depth |
| `max_calls_per_subagent` | `int` | `20` | Max LLM calls per subagent |
| `truncate_len` | `int` | `2000` | Output chars shown to the LLM per step |
| `max_money_spent` | `float` | `1.0` | Hard budget cap in USD |
| `max_completion_tokens` | `int` | `50000` | Max total completion tokens across all subagents |
| `max_prompt_tokens` | `int` | `200000` | Max total prompt tokens across all subagents |

## Best Practices & Troubleshooting

- **Place your task at the top or bottom of the prompt** — the REPL restricts how much context the LLM sees, so don't bury the task in the middle.
- **Mark structured data with backtick blocks** — wrap JSON, CSV, etc. in fenced code blocks and name the format in the prompt.
- **Use strong coding models** — agents write and execute Python, so coding benchmarks matter. See [recommended models](https://avbiswas.github.io/fast-rlm/guide/configuration/#model-names).
- **Inject domain docs when needed** — for obscure domains, add reference material and tell the agent how it's organized (e.g. with `##` headers).
- **Check logs and start with strict limits** — review what the agent is doing before scaling up. Prompt changes usually help more than bigger budgets.

For the full guide, see the [Best Practices & Troubleshooting](https://avbiswas.github.io/fast-rlm/guide/tips/) docs page.

## Log Viewer

![TUI Log Viewer](docs/images/tui.jpeg)

Every run saves a `.jsonl` log file to `logs/`.

```bash
# Print stats (no extra dependencies)
fast-rlm-log logs/run_xxx.jsonl

# Interactive TUI viewer (requires bun)
fast-rlm-log logs/run_xxx.jsonl --tui
```

---

## Development (from source)

### 1. Install Deno

Windows (npm):

```powershell
npm install -g deno
```

macOS / Linux:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

Then add Deno to your `PATH`:

```bash
export DENO_INSTALL="$HOME/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"
```

### 2. Install Bun (for the log viewer)

```bash
curl -fsSL https://bun.sh/install | bash
cd tui_log_viewer && bun install
```

### 3. API Key Setup

Set your key in `.env` or `.envrc`:

```bash
export RLM_MODEL_API_KEY=sk-or-...
```

### 4. Configuration

Edit `rlm_config.yaml` at the project root:

```yaml
max_calls_per_subagent: 20
max_depth: 3
truncate_len: 2000
primary_agent: "z-ai/glm-5"
sub_agent: "minimax/minimax-m2.5"
max_money_spent: 1.0
max_completion_tokens: 50000
max_prompt_tokens: 200000
```

### 5. Running

```bash
# Run the example
deno task test_counting_r

# Run the subagent directly
echo "What is 2+2?" | deno task subagent

# View logs
./viewlog logs/<logfile>.jsonl
```

### 6. Benchmarks

```bash
uv sync --extra benchmarks
uv run benchmarks/oolong_synth_benchmark.py
uv run benchmarks/longbench_benchmark.py
```

---

## Contributing

- **Small PRs only** — keep changes focused and minimal. Large PRs will not be accepted.
- **No LLM-generated slop** — AI-assisted code is fine, but bulk-generated boilerplate with no thought behind it will be rejected.
- **Minor features welcome** — small, well-scoped PRs that add useful functionality will be considered.
- **Large feature requests** — open an issue first to discuss the design before writing any code.

## License

MIT License. See [LICENSE](LICENSE).
