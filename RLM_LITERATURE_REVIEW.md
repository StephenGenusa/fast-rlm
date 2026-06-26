# RLM literature review, baseline feature set, and implementation audit

*Compiled mid-2026 from the primary paper/blog and the public implementation
ecosystem. Primary sources (arXiv abstract, the authors' blog) were read in full;
ecosystem feature claims come from project READMEs/sites via web search and should
be treated as a snapshot — verify against the linked repos before relying on
version-specific details.*

---

## Part 1 — Literature review

### 1.1 The canonical definition (Zhang, Kraska, Khattab)

A Recursive Language Model is *a thin scaffold around a base model `M` that keeps
the same interface as a single call* — `rlm.completion(query, context) -> str` is a
drop-in for `llm.completion(...)` — but changes what happens inside. The defining
mechanics, stated in the paper (arXiv:2512.24601) and blog:

1. **Context-as-environment, not prompt.** The (possibly huge) context `C` is
   loaded as a *variable* in an external environment; the **root LM** (depth 0) is
   given only the **query** `q` plus small metadata about `C`. The model never sees
   the whole context at once.
2. **Query/context separation.** The query is the only task text the root sees;
   context lives behind a variable. This separation is the mechanism, not an
   incidental detail, and it propagates to every recursion level.
3. **A REPL environment.** The canonical instantiation is a Jupyter-like Python
   REPL pre-loaded with `context`. The model interacts by emitting code blocks; the
   environment executes them and feeds back **truncated** stdout so the root's
   context grows slowly regardless of `|C|`.
4. **Recursive sub-LM calls.** Inside the REPL the model can call a sub-LM over a
   slice it selects — `RLM_M(q̂, Ĉ)` spawns an isolated instance. The base case
   (simplest environment) is an ordinary model call, so an RLM strictly generalizes
   a single call.
5. **Termination.** `FINAL(answer)` returns an inline answer; `FINAL_VAR(name)`
   returns the value of a REPL variable the model built up.
6. **Bounded recursion depth.** The paper restricts experiments to **depth 1** (the
   root calls leaf LMs, not other RLMs) and flags deeper recursion as future work.
7. **Emergent strategies** (observed, not hard-coded): *peek*, *grep*, *partition +
   map* (chunk and fire focused sub-calls), *summarize/reduce*, and *programmatic
   one-shot* (do the computation in code, skip the model — e.g. LoCoDiff diffs).
8. **Stated limitations.** The reference is **blocking / no prefix caching / no
   asynchrony** (seconds to minutes per query); no strong guarantees on total cost
   or runtime; depth-1 only; security of an arbitrary-code REPL is acknowledged.

Headline evidence: an RLM on the smaller **GPT-5-mini beat full GPT-5 by ~114%**
on the 132k OOLONG split at comparable cost; an RLM was the only method to hold
near-perfect accuracy at 1000 documents (~10M+ tokens) on BrowseComp-Plus; and a
first **natively recursive 8B model (RLM-Qwen3-8B)** was post-trained (+28.3% over
base), arguing the recursion trajectory is itself learnable/RL-able.

### 1.2 Derivative / critical literature

- **"Think, But Don't Overthink" (Wang, arXiv:2603.02615).** Reproduces depth-1
  gains on hard reasoning; shows **depth-2 "overthinking"** failures (parametric
  hallucination, role-play confusion, format collapse) and that wrapping *easy
  retrieval* in an RLM can be *worse* than a plain call. Engineering takeaway:
  recursion depth is a cost, not a free dial; don't RLM trivial tasks.
- **SRLM (Alizadeh et al., arXiv:2603.15653).** Program *selection* matters as much
  as execution; adds uncertainty-aware self-reflection (self-consistency, reasoning
  length, verbalized confidence) to choose among candidate REPL programs; up to ~22%
  over RLM under the same budget.

### 1.3 The implementation ecosystem (feature snapshot)

| Project | Sandbox(es) | Sub-calls | Tools / MCP | Notable extras |
|---|---|---|---|---|
| **alexzhang13/rlm** (official) | LocalREPL, **Docker**, **Modal**, cloud (e2b/daytona/prime per docs) | `llm_query` + `rlm_query` (rlm_query falls back to llm_query at `max_depth`) | — | inference **+ training** env (prime-rl/verifiers); multi-provider (OpenAI, Anthropic, OpenRouter, Portkey, LiteLLM); trajectory **visualizer**; `max_depth` (only 1 supported), `max_iterations` |
| **alexzhang13/rlm-minimal** | in-process `exec` | `llm_query` | — | gist-level reference; no cost tracking, no sandbox, blocking |
| **avbiswas/fast-rlm** | **Deno + Pyodide** (WASM) | `llm_query` + **`batch_llm_query`** (async parallel; model decides parallel/serial) | **user tools as plain Python fns** + **MCP servers** | JSON-**schema-validated** structured I/O; **compression-before-delegation**; sub-results returned as REPL variables (not auto-injected); **TUI trace viewer**; on PyPI |
| **Prime Intellect RLMEnv** | verifiers sandbox | recursive sub-LMs | tools exposed to **sub-LMs** | RL-training-oriented; "context folding"; model must **answer via an env variable** |
| **DSPy `RLM`** | sandboxed Python REPL | recursive sub-LLM | — | drop-in DSPy module integration |
| **grishahq/recursive-llm** | **RestrictedPython** | recursive | — | LiteLLM universal providers |
| **hampton-io/RLM** | Node/TS runtime | recursive | — | TypeScript port |
| **eesb99/rlm-mcp** | — | — | **exposes RLM *as* an MCP server** | verified code execution + LLM reasoning |
| **fullstackwebdev/rlm_repl** | local OpenAI-compatible | recursive | — | targets local GGUF (Qwen-Coder) |

**Convergent ecosystem lessons:** (1) keep query/context separate at every level;
(2) truncate REPL output at the scaffold; (3) return sub-results as REPL variables,
don't auto-inject; (4) parallelize sub-calls; (5) sandbox the REPL for untrusted
input; (6) pass a schema when you expect structured output.

---

## Part 2 — Baseline feature set

Two tiers: the **core contract** (required to *be* an RLM) and **common
ecosystem features** (present across mature implementations but not definitional).

### 2.1 Core RLM contract (must-have)

| # | Capability | Source |
|---|---|---|
| C1 | `complete(query, context) -> str` drop-in interface | paper §2 |
| C2 | Context stored as an environment **variable**, not in the prompt | paper §2 |
| C3 | Query/context **separation** at every level | paper §2.2 |
| C4 | Code-driven REPL the model explores (peek/grep/slice/transform) | paper §2.3 |
| C5 | **Recursive sub-LM calls** over selected slices | paper §2.1 |
| C6 | Scaffold **truncates** REPL output back to the root | paper §2.3 |
| C7 | `FINAL` / `FINAL_VAR` termination | paper §2.3 |
| C8 | Bounded **recursion depth** (default 1) | paper §2.4 |
| C9 | Sub-results returned as variables, **not auto-injected** | blog / fast-rlm |

### 2.2 Common ecosystem features (expected in a mature engine)

| # | Capability | Who has it |
|---|---|---|
| E1 | Real **sandbox** isolation (Docker/Modal/cloud or WASM) | official, fast-rlm |
| E2 | **Parallel** sub-calls (`batch_llm_query`) | fast-rlm |
| E3 | **Async** sub-calls | fast-rlm |
| E4 | **Multi-provider** clients (OpenAI/Anthropic/OpenRouter/LiteLLM…) | official |
| E5 | **Tools** in the REPL (user fns) | fast-rlm |
| E6 | **MCP** integration | fast-rlm; eesb99 (as server) |
| E7 | **Schema-validated** structured sub-call output | fast-rlm |
| E8 | **Compression-before-delegation** | fast-rlm |
| E9 | **Trajectory visualizer / logging** | official, fast-rlm |
| E10 | **Training / RL** environment | official, Prime Intellect |
| E11 | A **natively recursive** trained model | paper (RLM-Qwen3-8B) |
| E12 | **Cost/usage** accounting | (gap in rlm-minimal; varies) |

---

## Part 3 — Audit of *this* engine (`rlm_engine`)

### 3.1 (a) Compliance with the core RLM contract — **PASS (9/9)**

| # | Contract item | Status | Where |
|---|---|---|---|
| C1 | drop-in `complete(query, context)` | ✅ | `core.py RLM.complete` |
| C2 | context as REPL variable | ✅ | `set_context`; root prompt gets metadata only |
| C3 | query/context separation every level | ✅ | `prompts.py`; `llm_query(task, data=...)` keeps them apart; holds in recursion |
| C4 | code-driven REPL | ✅ | Pyodide/LocalREPL; AST notebook exec |
| C5 | recursive sub-LM calls | ✅ | `llm_query` / `batch_llm_query` |
| C6 | scaffold truncates output | ✅ | head 4000 / tail 1000, scaffold-side |
| C7 | `FINAL` / `FINAL_VAR` | ✅ | REPL fns + balanced-paren text fallback |
| C8 | bounded depth (default 1) | ✅ | `max_depth=1` default |
| C9 | sub-results as variables, not auto-injected | ✅ | results returned into the REPL, not the root prompt |

**Verdict: fully compliant** with the canonical RLM paradigm, and it implements the
structural query/context separation that the paper calls the core mechanism.

### 3.2 (b) Features in the literature we did **NOT** implement

Ranked by how much they matter for parity:

1. **Training / RL environment (E10) and a natively-recursive model (E11).** We are
   inference-only. No prime-rl/verifiers harness, no RLMEnv, no post-trained RLM
   model. This is the single biggest scope gap vs the official project + Prime
   Intellect — and the paper's central forward-looking claim.
2. **Docker / Modal / cloud sandboxes (part of E1).** We ship a hardened
   Deno/Pyodide WASM sandbox + an unsafe LocalREPL. We do **not** offer
   Docker/Modal/e2b/daytona/prime backends the official library has. (Our WASM
   sandbox is arguably stronger per-process isolation, but the deployment options
   are narrower.)
3. **Built-in MCP (E6).** We have a generic host tools channel; MCP works only if
   the user writes a proxy tool. fast-rlm connects to MCP servers natively, and
   there's a project exposing RLM *as* an MCP server — neither is built in here.
4. **Compression-before-delegation (E8).** fast-rlm compresses context before
   forwarding it a level deeper; we forward slices as-is.
5. **Trajectory visualizer / TUI (E9).** We record a `trajectory` list
   (iter/code/stdout) but ship **no viewer** (the official repo and fast-rlm both
   have one).
6. **Native `rlm_query` vs `llm_query` split.** The official API exposes both
   (`rlm_query` = recursive child, falling back to `llm_query` at depth). We expose
   one `llm_query` and route recursion internally by depth — functionally
   equivalent, naming differs.
7. **Native non-LiteLLM providers (part of E4).** The official lib has first-class
   OpenRouter/Portkey clients; we cover those only via `LiteLLMClient`.
8. **Enforced "answer via env variable" (RLMEnv).** We *support* `FINAL_VAR` but
   don't *require* returning through a variable the way RLMEnv does for training.
9. **DSPy/framework integration.** No DSPy module wrapper.

### 3.3 (c) Features we implemented that are **not** in the (published) literature

Some popular features (parallel/async sub-calls E2/E3, Pyodide+Deno sandbox,
user tools E5, schema validation E7, cost accounting E12) **are** in the ecosystem
(mostly fast-rlm/official) — so they are parity, **not** novel, and are excluded
here. The following are genuinely beyond what the literature/ecosystem documents:

1. **Per-cell execution timeout with kill + auto-restart.** A `exec_timeout` that
   measures *sandbox-compute* time (excluding sub-call latency) and kills/restarts a
   runaway interpreter (`while True`). The paper explicitly lacks runtime guarantees;
   no ecosystem project documents a compute-watchdog.
2. **Chunked context streaming** (`init_begin`/`ctx_chunk`/`init_end`) to move
   contexts past the single-stdio-line ceiling — others transfer context in one shot.
3. **Arbitrary Python object context via a pickle codec** (`context_codec`), with a
   clear class-availability error. The paper *speculates* "any modality loadable into
   memory" but ships str/JSON; we make non-JSON objects actually cross the boundary.
4. **Warm-sandbox pool** (`SandboxPool`, `RLM(sandbox_pool=)`) with health-check /
   replace and lightweight `reset()` — amortizes the cold start; not described
   elsewhere.
5. **Provider retry/backoff + per-call timeout** baked into the clients.
6. **Dollar-cost accounting** (`Usage.cost_usd` via LiteLLM price map / editable
   table) — most references track tokens at best.
7. **Concrete isolation hardening + a verified probe**: Deno read scoped to the
   cache dir, `--deny-env`, `--node-modules-dir=none`, **API-key scrubbing from the
   subprocess env**, throwaway temp cwd — with a documented allowed/denied probe.
   fast-rlm uses Deno/Pyodide too, but this specific lockdown + evidence goes beyond
   what's published.
8. **Lifecycle niceties**: idempotent `start()`, `reset()`, `clone()` for per-level
   recursion isolation, and a deterministic `MockClient` + fully offline test suite.

### 3.4 Notable design choice vs the paper

We treat **depth > 1 as supported** (each recursive sub-call gets its own fresh
sandbox), whereas the paper defaults to depth 1 and the reproduction study shows
depth-2 "overthinking." Our default is still `max_depth=1`; deeper is opt-in and
documented as slower and prone to overthinking — i.e. we enable it but steer users
to the paper's recommended default.

---

## Bottom line

The engine is a **faithful, fully-compliant RLM** (9/9 core contract) that matches
most mature-ecosystem features (sandboxing, parallel/async sub-calls, tools, schema
validation, multi-provider, cost) and adds a cluster of **production-hardening**
features not found in the literature (compute timeout, context streaming, pickle
contexts, warm pool, retries, key-scrubbing isolation). Its real gaps are **not**
about the RLM mechanism itself but about **scope**: no training/RL environment, no
native RLM model, fewer sandbox backends, and no built-in MCP/visualizer.

## Sources

- Zhang, Kraska, Khattab. *Recursive Language Models.* arXiv:2512.24601 — https://arxiv.org/abs/2512.24601 ; blog — https://alexzhang13.github.io/blog/2025/rlm/
- Official code — https://github.com/alexzhang13/rlm ; minimal — https://github.com/alexzhang13/rlm-minimal
- fast-rlm — https://github.com/avbiswas/fast-rlm ; site — https://avbiswas.github.io/fast-rlm/
- DSPy RLM module — https://dspy.ai/api/modules/RLM/
- Prime Intellect, *RLM: the paradigm of 2026* — https://www.primeintellect.ai/blog/rlm
- Wang, *Think, But Don't Overthink* — arXiv:2603.02615
- Alizadeh et al., *SRLM* — arXiv:2603.15653
- Other ports: grishahq/recursive-llm, hampton-io/RLM, eesb99/rlm-mcp, fullstackwebdev/rlm_repl
