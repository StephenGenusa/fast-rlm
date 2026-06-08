// Real-Pyodide test: explicitly call llm_query / batch_llm_query / asyncio.gather
// INSIDE the REPL and observe what happens. Stubs the JS bridges (__js_llm_query__,
// __js_batch_confirm__) so no real LLM is invoked. The Python below mirrors the
// query primitives + the gather failsafe from subagents.ts setup_code.
//
// Run:  deno run --allow-read --allow-env --allow-net --allow-write tests/repl_calls_test.ts
import { loadPyodide } from "pyodide";

// ---- the primitives, exactly as injected by the engine (guard ON) ----
const PRIMITIVES = `
class _LazyQuery:
    __slots__ = ("context", "schema", "tools", "mcp")
    def __init__(self, context, schema, tools, mcp):
        self.context = context; self.schema = schema; self.tools = tools; self.mcp = mcp
    def __await__(self):
        return self._run(False).__await__()
    async def _run(self, suppress):
        _result = await __js_llm_query__(self.context, self.schema, None, None, suppress)
        if hasattr(_result, "to_py"):
            return _result.to_py()
        return _result

def llm_query(context, schema=None, *, tools=None, mcp=None):
    return _LazyQuery(context, schema, tools, mcp)

async def batch_llm_query(*queries):
    import asyncio as _asyncio
    import json as _json
    qs = list(queries)
    if len(qs) == 1 and isinstance(qs[0], (list, tuple)):
        qs = list(qs[0])
    if not all(isinstance(q, _LazyQuery) for q in qs):
        raise TypeError("batch_llm_query expects llm_query(...) calls")
    def _meta(ctx):
        s = ctx if isinstance(ctx, str) else _json.dumps(ctx)
        return {"childChars": len(s), "preview": s[:160]}
    _parent = context if isinstance(context, str) else _json.dumps(context)
    _payload = {"parentChars": len(_parent), "items": [_meta(q.context) for q in qs]}
    _approved = await __js_batch_confirm__(_json.dumps(_payload))
    if not _approved:
        raise RuntimeError("BATCH_DELEGATION_REJECTED: under-compressed batch")
    return await _asyncio.gather(*[q._run(True) for q in qs])

import asyncio as _aio_guard
_real_gather = _aio_guard.gather
def _guarded_gather(*aws, **kw):
    if any(isinstance(a, _LazyQuery) for a in aws):
        raise RuntimeError(
            "Do not call llm_query inside asyncio.gather. Use batch_llm_query(...) instead.")
    return _real_gather(*aws, **kw)
_aio_guard.gather = _guarded_gather
`;

const log: { llmCalls: { suppress: boolean }[]; batchCalls: unknown[] } = {
    llmCalls: [],
    batchCalls: [],
};

let fail = false;
function ck(label: string, cond: boolean, extra = "") {
    console.log(`  ${cond ? "✔" : "✗"} ${label}${extra ? "  " + extra : ""}`);
    if (!cond) fail = true;
}

const pyodide = await loadPyodide();

// stub bridges
pyodide.globals.set("__js_llm_query__", async (ctx: unknown, _s: unknown, _t: unknown, _m: unknown, suppress: unknown) => {
    log.llmCalls.push({ suppress: Boolean(suppress) });
    return `r::${String(ctx)}`;
});
pyodide.globals.set("__js_batch_confirm__", async (metaJson: unknown) => {
    log.batchCalls.push(JSON.parse(String(metaJson)));
    return true; // approve
});
pyodide.globals.set("context", "X".repeat(12000)); // parent context

await pyodide.runPythonAsync(PRIMITIVES);

async function attempt(code: string): Promise<{ ok: boolean; value?: unknown; err?: string }> {
    try {
        const value = await pyodide.runPythonAsync(code);
        return { ok: true, value: value?.toJs ? value.toJs() : value };
    } catch (e) {
        return { ok: false, err: e instanceof Error ? e.message : String(e) };
    }
}

console.log("\n[A] await llm_query('ctxA')  — single call, should run");
log.llmCalls.length = 0;
let r = await attempt(`await llm_query("ctxA")`);
ck("returns a result", r.ok && String(r.value) === "r::ctxA", `value=${r.value}`);
ck("ran with suppress=False (per-call guard active)", log.llmCalls.at(-1)?.suppress === false);

console.log("\n[B] await asyncio.gather(llm_query(a), llm_query(b))  — should be BLOCKED");
r = await attempt(`import asyncio\nawait asyncio.gather(llm_query("a"), llm_query("b"))`);
ck("raised (blocked)", !r.ok);
ck("error steers to batch_llm_query", !!r.err && r.err.includes("batch_llm_query"), r.err?.split("\n").pop() ?? "");

console.log("\n[C] await asyncio.gather(*[llm_query(c) for c in ...])  — should be BLOCKED");
r = await attempt(`import asyncio\nawait asyncio.gather(*[llm_query(c) for c in ["c1","c2","c3"]])`);
ck("raised (blocked, comprehension form)", !r.ok && !!r.err && r.err.includes("batch_llm_query"));

console.log("\n[D] await batch_llm_query(llm_query(b1), llm_query(b2))  — should run, ONE judge");
log.llmCalls.length = 0;
log.batchCalls.length = 0;
r = await attempt(`await batch_llm_query(llm_query("b1"), llm_query("b2"))`);
ck("returns results in order", r.ok && JSON.stringify(r.value) === JSON.stringify(["r::b1", "r::b2"]), `value=${JSON.stringify(r.value)}`);
ck("exactly ONE batch-confirm call", log.batchCalls.length === 1);
ck("children ran suppressed", log.llmCalls.length === 2 && log.llmCalls.every((c) => c.suppress === true));

console.log("\n[E] await asyncio.gather(other(), other())  — non-llm_query, should NOT be blocked");
r = await attempt(`
import asyncio
async def other():
    return 1
await asyncio.gather(other(), other())
`);
ck("works (real coroutines pass through)", r.ok && JSON.stringify(r.value) === JSON.stringify([1, 1]), `value=${JSON.stringify(r.value)}`);

console.log("\n[F] await batch_llm_query([llm_query(x), llm_query(y)])  — list form");
r = await attempt(`await batch_llm_query([llm_query("x"), llm_query("y")])`);
ck("list form works", r.ok && JSON.stringify(r.value) === JSON.stringify(["r::x", "r::y"]));

console.log("\n" + "=".repeat(48));
if (fail) {
    console.log("REPL CALL TESTS FAILED");
    Deno.exit(1);
}
console.log("ALL REPL CALL TESTS PASSED (no LLM invoked)");
