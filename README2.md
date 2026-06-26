# rlm-engine

A working **Recursive Language Model** engine with a **mandatory code sandbox**.
An RLM treats a (potentially huge) context as a *variable in a REPL* instead of
stuffing it into the prompt; the model writes code to inspect, slice, search, and
recursively sub-query it. So `rlm.complete(query, context)` is a drop-in for a
single LLM call but works far beyond the context window and resists context rot.

Based on Zhang, Kraska & Khattab, *Recursive Language Models* (arXiv:2512.24601).

## Why a sandbox is mandatory

The root model **writes the Python that runs**. That code is untrusted, so it
executes as CPython-in-WASM (Pyodide) inside a separate, locked-down process with
**no network and no API keys**. Its only exit is the `llm_query`/`batch_llm_query`
channel, which the host controls. Under **Deno** (the production default) the
sandbox process runs with net/write/run/ffi **and env** denied and filesystem
**read scoped to the Deno cache dir** only, so even a WASM escape cannot read host
files/secrets, reach the network, write, or spawn. Provider API keys are scrubbed
from the subprocess environment on both runtimes. Full threat model:
[`docs/sandbox.md`](docs/sandbox.md).

## The bug this fixes

Naive "recursive prompt" attempts concatenate the query and the long text into
one string, so the model can't tell its task from its data. RLMs fix this
*structurally*: the query is the only task text the root model sees; the context
lives behind a `context` variable; sub-calls pass task and data as separate
arguments. The engine enforces that at every level of recursion.

## Install

```bash
pip install -e ".[litellm]"          # provider-agnostic client (OpenRouter, local, hosted)
# sandbox runtime — pick one:
npm install -g deno && deno cache npm:pyodide   # recommended (hardened)
# or, dev only:
npm install pyodide                  # uses Node (weaker isolation; see docs/sandbox.md)
```

## Quickstart (OpenRouter)

```python
from rlm import RLM, LiteLLMClient, PyodideSandbox

rlm = RLM(
    client     = LiteLLMClient("openrouter/anthropic/claude-3.5-sonnet"),  # root
    sub_client = LiteLLMClient("openrouter/anthropic/claude-3.5-haiku"),   # sub-calls
    sandbox    = PyodideSandbox(runtime="auto"),   # Deno preferred, Node fallback
    max_depth  = 1,    # depth>1 tends to "overthink" (arXiv:2603.02615)
)
out = rlm.complete(query="What is the magic number?", context=huge_string)
print(out.answer, out.usage)
```

Works with `AnthropicClient`, `OpenAIClient`, or any object implementing
`complete(messages) -> Completion`.

## Command line

After `pip install -e ".[litellm,tui]"` two commands are available:

```bash
# run a query over a context file (model from --model or $RLM_MODEL)
rlm -m openrouter/anthropic/claude-3.5-sonnet "What is the magic number?" context.txt
rlm -m gpt-4o-mini --trace-content --show-usage "summarize" big.txt   # autosaves the trace
cat big.txt | rlm -m gpt-4o-mini "summarize" -            # context from stdin

# view a saved trace, or run one live, in the Textual TUI
rlm-tui rlm_traces/<trace_id>.jsonl
rlm-tui --live -m gpt-4o-mini --trace-content "summarize" big.txt
```

Common flags: `--sub-model`, `--api-base/--api-key` (local servers), `--max-depth`,
`--max-iterations`, `--runtime {auto,deno,node}`, `--allow-unsafe-node`,
`--trace-content`, `--trace-dir`, `--json` (parse the context file as JSON).
(Equivalent module forms: `python -m rlm.cli …`, `python -m rlm.tui file.jsonl`.)

## Local models

vLLM, llama.cpp, Ollama, and LM Studio all expose OpenAI-compatible endpoints, so
one `LiteLLMClient` config drives any of them. Because RLMs fan out parallel
sub-calls, **serving throughput matters more than single-stream latency** — see
[`docs/local-models.md`](docs/local-models.md) for the vLLM-vs-GGUF tradeoff and
concurrency tuning. Quick version:

```python
client = LiteLLMClient("openai/Qwen2.5-14B-Instruct",
                       api_base="http://localhost:8000/v1", api_key="sk-noop")
RLM(client=client, sub_client=client, sub_max_parallel=8, sandbox=PyodideSandbox())
```

## What the model can do in the sandbox

| Call | Purpose |
|---|---|
| `print(context[:2000])` | peek at slices (output truncated by the scaffold) |
| `await llm_query(task, data=None, schema=None)` | one focused **leaf** sub-call; task and data kept separate; `schema` validates/parses JSON |
| `await batch_llm_query(jobs)` | run many leaf sub-calls **in parallel**, results in order |
| `await rlm_query(task, data=…, schema=None)` | spawn a **recursive child RLM** over `data` (only when `max_depth>1`; otherwise a leaf) |
| `await batch_rlm_query(jobs)` | run many recursive children in parallel |
| `FINAL(value)` | end the run, return any value (str/dict/list/number) |


## Backend

`PyodideSandbox(runtime="deno"|"node"|"auto")` is the only execution backend — the
hardened WASM sandbox. There is no in-process `exec` backend: running
model-generated code in your own process is unsafe by construction, so it is not
offered. Use `SandboxPool` to amortize sandbox startup across calls.

## Tests

Offline suites need no API key or JS runtime (mock model + a scripted sandbox);
`test_sandbox.py` needs Deno/Node + pyodide.

```bash
# offline (pure logic, clients via mocked SDKs, complete() via scripted sandbox)
for t in clients core_units prompts tracing sandbox_units tui cli rlm; do
  python tests/test_$t.py; done
python tests/test_sandbox.py   # real Pyodide WASM sandbox (deno/node + pyodide)

# coverage (pip install -e ".[dev]")
python -m coverage run -p --source=rlm tests/test_rlm.py      # repeat per file
python -m coverage combine && python -m coverage report -m
```

Real tests by default; mocks only at true seams — provider SDKs (to avoid the
network) and a `ScriptedSandbox` for offline `complete()` branch coverage. The
real sandbox's isolation, timeouts, recursion, streaming, pickle, tools, and
key-scrubbing are exercised against an actual Deno/Pyodide subprocess in
`test_sandbox.py`. Combined coverage ≈ **91%** (pure modules 95–100%).

## Design notes over the canonical minimal reference

Mandatory WASM sandbox (Deno: scoped read, denied env/net/write/run/ffi, scrubbed
keys) · strict query/context separation at every depth · parallel sub-calls · AST
notebook execution with top-level `await` · robust FINAL (REPL function +
balanced-paren text fallback) · per-cell execution timeout with auto-restart ·
namespace reset on reuse · isolated recursion (`max_depth>1` spawns a fresh
sandbox per sub-call) · `jsonschema` validation + repair for `schema=` · provider
retry/backoff/timeout · token + dollar-cost accounting across root + all sub-calls ·
context streaming (chunked init) for large inputs · host tools channel
(`RLM(tools=...)` → `await tool(...)`) · arbitrary context objects via a pickle
codec (`context_codec=`) · warm-sandbox pooling (`SandboxPool` /
`RLM(sandbox_pool=)`) · per-call tracing/observability (`RLMResult.trace`,
`RLM(on_span=…)`, OTel-shaped spans — see `docs/observability.md`) ·
provider-agnostic via LiteLLM.

Node has no OS permission layer and requires `allow_unsafe_node=True`. Schema
validation needs the `schema` extra (`pip install -e ".[schema]"`).
