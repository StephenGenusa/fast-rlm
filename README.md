# Recursive Language Models (RLM)

A research synthesis plus a working, **sandboxed** Recursive Language Model
engine.

*This is a work in progress/experimental project because I wanted more than
the original project offered.*

An RLM answers a query about a context by treating that context as a **variable
in a code REPL** instead of stuffing it into the prompt. The root model never
sees the raw text; it writes Python to inspect, slice, search, and recursively
sub-query the context, then returns a final answer. The result is a near
drop-in replacement for a single LLM call (`rlm.complete(query, context)`) that
works far beyond the context window and resists context rot.

Based on Zhang, Kraska & Khattab, *Recursive Language Models* (arXiv:2512.24601).

## The bug this fixes

Naive "recursive prompt" attempts concatenate the query and the long text into
one string, so the model cannot tell its instructions from its data and gets
confused. RLM removes that ambiguity **structurally**: the query is the only
task text the root model sees; the context lives behind a `context` variable;
sub-calls pass task and data as separate arguments. That invariant holds at every
level, regardless of file size — a 2 MB file never enters a prompt.

## Repository layout

```
RLM/
├── README.md                                  # this file
├── Recursive_Language_Models_Research_Synthesis.md   # the research paper/synthesis
└── rlm_engine/                                # the working engine (pip-installable)
    ├── README.md                              # engine-specific quickstart
    ├── pyproject.toml
    ├── package.json                           # declares the pyodide npm dependency
    ├── rlm/
    │   ├── __init__.py                         # public API (v0.2.0)
    │   ├── core.py                             # RLM orchestrator + sub-call routing
    │   ├── tracing.py                          # Span/Tracer — per-call observability
    │   ├── tui.py                              # Textual trace viewer (run_tui/live_tui)
    │   ├── cli.py                              # `rlm` and `rlm-tui` console scripts
    │   ├── sandbox.py                          # PyodideSandbox (hardened WASM sandbox)
    │   ├── sandbox_host.mjs                    # Deno/Node host that runs Pyodide
    │   ├── clients.py                          # Anthropic / OpenAI / LiteLLM / Mock clients
    │   └── prompts.py                          # root-model system prompt + step prompts
    ├── docs/
    │   ├── sandbox.md                          # sandbox threat model & setup
    │   └── local-models.md                     # vLLM / llama.cpp / Ollama / LM Studio
    ├── examples/
    │   ├── niah_demo.py                        # needle-in-a-haystack
    │   ├── long_doc_qa.py                      # long-document QA
    │   └── local_vllm_demo.py                  # fully local, sandboxed run
    └── tests/
        ├── test_rlm.py                         # offline (mock model, no runtime needed)
        └── test_sandbox.py                     # real Pyodide WASM sandbox
```

## Why a sandbox is mandatory

The root model **writes the Python that runs**. That code is untrusted by
construction, so it executes as CPython-in-WASM (Pyodide) inside a separate
process with **no network and no API keys**. Its only exit is the
`llm_query`/`batch_llm_query` channel, which the host controls and uses to make
the real API calls. Under **Deno** (the production default) the sandbox process
runs with net/write/run/ffi **and env** denied, and filesystem **read scoped to
the Deno cache dir** only — so even a WASM escape lands in a process that cannot
read host files or secrets, reach the network, write, or spawn. On both runtimes
the subprocess environment is scrubbed of provider API keys (the host process
makes the real calls). Full threat model in `rlm_engine/docs/sandbox.md`.

---

## Setup

### Requirements

- **Python ≥ 3.10**
- A **JS runtime for the sandbox**: Deno (recommended) *or* Node ≥ 18, plus the
  `pyodide` npm package.
- API credentials for whatever model provider you use (or a local server).

### 1. Install the Python package

```bash
cd rlm_engine
pip install -e ".[litellm]"      # LiteLLM client: OpenRouter + local + hosted
# or pick a specific provider extra:
#   pip install -e ".[anthropic]"
#   pip install -e ".[openai]"
```

### 2. Install a sandbox runtime

Recommended — Deno (hardened: WASM boundary + denied OS permissions):

```bash
curl -fsSL https://deno.land/install.sh | sh    # or: npm i -g deno
deno cache npm:pyodide                           # pre-fetch so runs need no network
```

Dev fallback — Node (works, but weaker isolation; see `docs/sandbox.md`):

```bash
cd rlm_engine
npm install pyodide
```

### 3. Provide credentials

```bash
export OPENROUTER_API_KEY=...      # for OpenRouter via LiteLLM
# or ANTHROPIC_API_KEY / OPENAI_API_KEY for the native clients
# local servers (vLLM/llama.cpp/Ollama/LM Studio) need no real key
```

---

## Usage

### Quickstart (OpenRouter)

```python
from rlm import RLM, LiteLLMClient, PyodideSandbox

rlm = RLM(
    client     = LiteLLMClient("openrouter/anthropic/claude-3.5-sonnet"),  # root
    sub_client = LiteLLMClient("openrouter/anthropic/claude-3.5-haiku"),   # sub-calls
    sandbox    = PyodideSandbox(runtime="auto"),   # Deno preferred, Node fallback
    max_depth  = 1,
)

out = rlm.complete(query="What is the magic number?", context=huge_string)
print(out.answer)
print(out.usage)        # token counts across root + every sub-call
```

If you omit `sandbox=...`, a `PyodideSandbox` is created and torn down per call.

### Local models

vLLM, llama.cpp, Ollama, and LM Studio all expose OpenAI-compatible endpoints, so
one `LiteLLMClient` config drives any of them. Because RLMs fan out **parallel**
sub-calls, serving throughput matters more than single-stream latency — see
`docs/local-models.md` for the vLLM-vs-GGUF tradeoff and concurrency tuning.

```python
client = LiteLLMClient("openai/Qwen2.5-14B-Instruct",
                       api_base="http://localhost:8000/v1", api_key="sk-noop")
rlm = RLM(client=client, sub_client=client,
          sub_max_parallel=8,            # match your server's concurrent slots
          sandbox=PyodideSandbox())
```

### What the model can do inside the sandbox

| Call | Purpose |
|---|---|
| `print(context[:2000])` | peek at slices (stdout is truncated by the scaffold) |
| `await llm_query(task, data=None, schema=None)` | one focused sub-call; task and data kept separate |
| `await batch_llm_query(jobs)` | run many leaf sub-calls **in parallel**, results in order |
| `await rlm_query(task, data=…)` | spawn a **recursive child RLM** over `data` (only when `max_depth>1`; else a leaf) |
| `await batch_rlm_query(jobs)` | run many recursive children in parallel |
| `FINAL(value)` | end the run, return any value (str/dict/list/number) |
| `FINAL_VAR("name")` | end the run, return the REPL variable named `name` |

### Backend

**`PyodideSandbox(runtime="deno"|"node"|"auto")`** is the only execution backend —
the hardened WASM sandbox (Deno preferred). There is deliberately **no in-process
`exec` backend**: running untrusted model-generated code in your own process is
unacceptable for an RLM, so it isn't offered. Pair it with `SandboxPool` to skip
the cold start across calls.

### Key parameters (`RLM(...)`)

| Param | Default | Meaning |
|---|---|---|
| `client` | — | root model (writes the code) |
| `sub_client` | = `client` | model used for `llm_query`/`batch_llm_query` |
| `sandbox` | `PyodideSandbox()` | execution backend |
| `max_iterations` | `12` | max root REPL turns before a forced final |
| `max_depth` | `1` | recursion depth; `>1` spawns a fresh isolated sandbox per sub-call (slower) |
| `sub_max_parallel` | `8` | parallel workers for `batch_llm_query` |
| `verbose` | `False` | print each root turn |

`RLM(...)` also takes `on_span` (a callback fired per span, for live tracing/TUI),
`trace_content` (default `False`; set `True` to capture prompt/response text), and
`redact` (a scrubber applied to captured text). The run's full span tree is on
`RLMResult.trace`.

`PyodideSandbox(...)` adds: `exec_timeout` (default `120` s of sandbox compute
per cell; a runaway cell is killed and the sandbox auto-restarts),
`startup_timeout` (default `120` s), and `allow_unsafe_node` (default `False` —
required to use the non-isolating Node runtime).

### Run the examples

```bash
cd rlm_engine
python examples/niah_demo.py          # needs a provider key + sandbox runtime
python examples/local_vllm_demo.py    # set RLM_API_BASE / RLM_MODEL first
```

### Run the tests

```bash
cd rlm_engine
python tests/test_rlm.py        # offline: mock model, no API key, no runtime
python tests/test_sandbox.py    # real Pyodide WASM (needs deno/node + pyodide)
```

`test_sandbox.py` proves isolation (host filesystem invisible, `import js`
blocked), the async `llm_query` bridge, parallel `batch_llm_query`, `FINAL`
round-trips, and a full RLM run driving the real WASM sandbox to the answer.

---

## How it works (brief)

1. The context is loaded as a `context` variable inside the sandbox; the root
   prompt gets the query plus *metadata* (type, size) only.
2. The root model writes ` ```repl ` code blocks. The engine executes each block
   in a persistent notebook namespace and feeds back the (truncated) stdout.
3. Code can call `llm_query`/`batch_llm_query`; those marshal out to the host,
   which runs the sub-model(s) — in parallel for batches — and returns results.
4. The model calls `FINAL(...)` to end the run. Token/usage is aggregated across
   the root and all sub-calls.

Scaffold truncation (head 4000 + tail 1000 chars by default) bounds what any
REPL output can inject back into the root context, so file size never inflates
the root prompt.

---

## Recently fixed

- **Execution timeout.** A per-cell `exec_timeout` (default 120 s of sandbox
  compute, excluding sub-call time) kills a runaway interpreter and auto-restarts
  it; the cell returns an `ExecutionTimeout` message.
- **Recursion (`max_depth ≥ 2`).** Each recursive sub-call now runs in its **own
  fresh sandbox** (isolated context, globals, and stdio pipe — safe under parallel
  batches). It costs one sandbox spawn per sub-call, so it's slower; depth 1
  remains the default.
- **State leak / `reset()`.** Re-`set_context` now clears the REPL namespace, and
  `sandbox.reset()` clears it explicitly — reusing a sandbox no longer bleeds
  variables across queries.
- **Schema validation.** `llm_query(schema=...)` now validates with `jsonschema`
  (install the `schema` extra) and does one repair re-ask on failure. Without
  `jsonschema` installed it falls back to parse-only.
- **Provider retry.** Anthropic/OpenAI/LiteLLM calls use bounded retry with
  jittered backoff and a per-call timeout.
- **Dollar-cost accounting.** `Usage.cost_usd` is computed per call (via LiteLLM's
  price map when installed, else an editable `rlm.clients._PRICES` table) and
  aggregated across the root and all sub-calls; unknown models report `0` with a
  one-time warning.
- **Context streaming.** Contexts above ~256 KB are chunked over stdio
  (`init_begin`/`ctx_chunk`/`init_end`) instead of one giant JSON line, raising the
  practical ceiling to tens of MB. Note: Pyodide still holds the full string in
  WASM memory — this is chunked transfer, not zero-copy; true file-backed access
  would break Deno's read-scope and is out of scope.
- **Host tools channel.** Pass `RLM(tools={"name": callable, ...})`; model code
  calls `await tool("name", ...)` in the sandbox, the host runs the allowlisted
  callable (in the trusted host process) and returns a JSON result. Tool names and
  docstrings are advertised in the system prompt; unknown names error cleanly.
  Isolation is unchanged (only allowlisted callables run); MCP can be layered on as
  a tool that proxies to a server.
- **Isolation hardening.** Deno read is scoped to the cache dir, env is denied,
  and provider keys are scrubbed from the subprocess env; Node requires
  `allow_unsafe_node=True`. (See `rlm_engine/FIXES.md` for the probe evidence.)
- **Arbitrary context objects.** `context` can be any Python object.
  `PyodideSandbox(context_codec="auto"|"json"|"pickle")` pickles non-JSON objects
  across the boundary (default `auto`). The object's classes must be importable
  inside the sandbox, else a clear error; don't
  use the pickle codec for context whose bytes are untrusted (use `json`).
- **Warm-sandbox pooling.** `SandboxPool(size=N)` keeps warm subprocesses;
  `RLM(sandbox_pool=pool)` leases one per call to skip the ~5 s cold start. Dead
  subprocesses are detected and replaced; each sandbox is cleared between leases.
- **Observability / tracing.** Every model interaction (root turns, REPL cells,
  sub-calls, tool calls, schema repairs, recursive children) is recorded as a
  nested `Span` with timing/tokens/cost/depth/parent. Read `RLMResult.trace`, or
  stream live via `RLM(on_span=…)`; `Span.to_otel()` gives an OpenTelemetry-GenAI
  shape. Prompt/response text is **opt-in** (`trace_content=True`) with a `redact=`
  hook — off by default since context may hold secrets. Replaces the old flat
  `trajectory`. See `rlm_engine/docs/observability.md`.

## Known limitations

- **Sub-call `data` cap.** Non-string `data` passed to `llm_query` is capped at
  500,000 chars when serialized; string `data` is passed whole.
- **Context-class availability (pickle codec).** An arbitrary object sent to the
  Pyodide sandbox needs its classes importable there; a host-only `__main__` class
  can't be reconstructed — load the class source into the sandbox, or pass a
  JSON-serializable context.

---

## References

- Zhang, Kraska, Khattab. *Recursive Language Models.* arXiv:2512.24601 —
  https://arxiv.org/abs/2512.24601 · blog: https://alexzhang13.github.io/blog/2025/rlm/
- Official code: https://github.com/alexzhang13/rlm · minimal: `rlm-minimal`
- In-repo: `Recursive_Language_Models_Research_Synthesis.md`,
  `rlm_engine/RLM_LITERATURE_REVIEW.md`, `rlm_engine/FIXES.md`.

## License

MIT.
