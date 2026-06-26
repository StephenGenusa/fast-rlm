# Sandboxing model-generated code

An RLM's root model **writes Python that the engine executes**. That code is
untrusted by construction — it is generated from a context you may not control.
So a sandbox is not optional. The job of the sandbox is to make sure that, no
matter what code the model writes, it cannot reach the network, read or write
your files, or see your API keys. The only capability it is granted is to *ask
the host to run an LLM sub-call* through a controlled channel.

## The two-process design

```
  your host process (Python)            sandbox process (Deno/Node + Pyodide)
  ┌───────────────────────────┐         ┌──────────────────────────────────┐
  │ RLM loop                  │  stdio  │ CPython compiled to WASM          │
  │ holds API keys + network  │ <─────> │ runs model code in a WASM boundary│
  │ services llm_query/batch  │  JSON   │ context lives here as a variable  │
  └───────────────────────────┘         │ NO network, NO keys               │
                                         └──────────────────────────────────┘
```

- The **context** is loaded as a variable *inside* the sandbox; it never enters a
  prompt.
- Model code runs as **CPython-in-WASM (Pyodide)**. The WASM boundary means
  `os`, `open`, etc. operate on an in-memory virtual filesystem, not your disk.
- When the model calls `await llm_query(...)`, that marshals out as a JSON
  request to the host, which makes the real API call (with your keys, over the
  network) and passes only the result back in. A sandbox escape still yields no
  keys and no network.

## Why Deno is the production default

Pyodide alone is a WASM boundary, but Pyodide exposes an `import js` bridge to the
host JavaScript runtime. On **Node**, that host scope has `require`/`process`, so
a determined escape could reach the OS — and Pyodide's network polyfills can also
leave the sandbox. The engine adds a Python-side guard that strips the `js`
importer, but in-process guards are a secondary layer, not a boundary.

**Deno closes this at the runtime layer.** Launched with denied net/write/run/ffi
**and env**, and with filesystem **read scoped to the Deno cache dir** (plus a
private temp working directory), even a full WASM escape lands in a process the OS
will not let open a socket, write a file, spawn anything, read your files, or read
your environment. WASM boundary + denied permissions = defense in depth. The
engine runs Deno with:

```
deno run --node-modules-dir=none --allow-read=$DENO_DIR --deny-env \
         --deny-net --deny-write --deny-run --deny-ffi  sandbox_host.mjs
```

`--allow-read` is scoped to the Pyodide cache only (an unrestricted `--allow-read`
would let an escape read any user-readable file); `--deny-env` blocks env access;
`--node-modules-dir=none` forces Pyodide to load from the cache so the read scope
is knowable. Separately, the engine scrubs provider API keys from the subprocess
environment on **both** runtimes — the host process makes the real LLM calls, so
the sandbox never needs a key.

## Setup

Pyodide is pinned (`package.json` → `pyodide` `314.0.0`). Pre-fetch it once so the
sandbox runs with no network at runtime; `scripts/prefetch.sh` does this for
whichever runtime you have.

**Deno (recommended):**
```bash
curl -fsSL https://deno.land/install.sh | sh     # or: npm i -g deno
bash scripts/prefetch.sh                          # caches npm:pyodide@314.0.0
# (equivalently: deno cache npm:pyodide@314.0.0)
```

**Node (dev/testing only):**
```bash
npm install                                       # installs the pinned pyodide
```

**Air-gapped:** pre-fetch on a connected machine, copy the resulting `DENO_DIR`
(printed by `prefetch.sh`) to the target, and set `DENO_DIR` there. With the cache
present, `--deny-net` does not block startup. If the cache is missing, the sandbox
raises a clear error pointing back here.

Then:
```python
from rlm import RLM, PyodideSandbox, LiteLLMClient

rlm = RLM(
    client=LiteLLMClient("openrouter/anthropic/claude-3.5-sonnet"),
    sub_client=LiteLLMClient("openrouter/anthropic/claude-3.5-haiku"),
    sandbox=PyodideSandbox(runtime="deno"),   # or "auto" / "node"
)
out = rlm.complete(query="...", context=huge_text)
```

`runtime="auto"` prefers Deno. Falling back to Node requires
`allow_unsafe_node=True` (Node has no OS permission layer, so containment is only
the WASM boundary); without it, selecting Node raises.

## What runs where

| Capability | Inside sandbox (model code) | Host (engine) |
|---|---|---|
| Read/transform `context` | yes (it's a local variable) | — |
| Network / API keys | **no** | yes |
| Filesystem | virtual WASM FS only | real FS |
| `llm_query` / `batch_llm_query` | requests it via the bridge | performs the call |
| `tool(name, ...)` (if `tools=` set) | requests it via the bridge | runs the allowlisted callable |

## Host tools

Pass `RLM(tools={"name": callable, ...})` to expose host functions to model code as
`await tool("name", *args, **kwargs)`. The call marshals over the same controlled
channel as `llm_query`; only the **allowlisted** callables run, and they run in the
trusted host process (so tools may use the network/keys you give them — the sandbox
itself still cannot). Results must be JSON-serializable (non-serializable returns
are stringified). Unknown tool names return an error string rather than raising.
The tool name is positional-only, so a tool may itself take a `name=` argument. MCP
can be layered on as a tool that proxies to an MCP server.

## The unsafe escape hatch: `LocalREPL`

`LocalREPL(allow_unsafe=True)` runs model code with `exec` **in your Python
process**. It is fast and dependency-free and useful for tests or for code you
fully trust over data you fully trust — but it is **not** a security boundary.
The constructor refuses to build without `allow_unsafe=True`. Never point it at
untrusted context.

## Limitations

- The context is transferred at init over stdio: in one message when small, and
  **chunked** (`init_begin`/`ctx_chunk`/`init_end`) above ~256 KB, which lifts the
  practical ceiling to tens of MB. Pyodide still holds the whole string in WASM
  memory, so this is chunked transfer, not zero-copy streaming; true file-backed
  access would require mounting a host file into MEMFS and is intentionally not
  done (it would break the Deno read-scope).
- **Context types / codecs.** `str` and JSON-serializable `dict`/`list` cross as
  text/JSON. Any other object crosses via `context_codec="pickle"` (default
  `"auto"` picks it automatically): the host base64-pickles it and the sandbox
  unpickles it. This needs the object's classes to be importable inside Pyodide —
  a class defined only in the host `__main__` cannot be reconstructed and yields a
  clear error. Pickled bytes come from the trusted host and run Deno-confined, but
  do not pickle context whose bytes are themselves untrusted (use `json`).
  `LocalREPL` runs in-process and accepts any live object directly.

## Warm-sandbox pooling

Starting Pyodide costs a few seconds. `SandboxPool(factory, size)` keeps warm
subprocesses and leases one per call, so that cost is paid once and amortized:

```python
from rlm import RLM, SandboxPool, PyodideSandbox
pool = SandboxPool(factory=lambda: PyodideSandbox(), size=4)
rlm = RLM(client=..., sandbox_pool=pool)   # complete() acquires/releases per call
...
pool.close()
```

One sandbox is leased to one caller at a time (a sandbox is a single stdio pipe and
is not concurrency-safe); `size` concurrent callers get distinct sandboxes. Dead
subprocesses are detected and replaced on acquire, and each sandbox's state is
cleared (`reset()`) before it returns to the pool.
- Pyodide ships a subset of the Python ecosystem; pure-Python and many scientific
  packages load via `micropip`, but arbitrary native wheels do not.
- The Python-side `import js` guard is best-effort; rely on Deno permissions for
  the actual guarantee.
