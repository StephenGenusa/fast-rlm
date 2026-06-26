"""
Real-sandbox tests — execute model code as CPython-in-WASM via Pyodide.

These spin up an actual Deno/Node + Pyodide subprocess, so they need a JS runtime
and the `pyodide` npm package. If neither is available the suite skips cleanly.

What they prove:
- code runs in the WASM boundary, not in this process;
- `context` is a real variable inside the sandbox;
- `await llm_query(...)` bridges out to a host handler and back;
- `batch_llm_query` returns results in order;
- `FINAL(...)` terminates and round-trips the value (incl. dicts);
- isolation holds: the host filesystem is NOT visible and `import js` is blocked.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm import PyodideSandbox, RLM, MockClient, SandboxPool  # noqa: E402


class _Unsendable:  # module-level -> pickles by reference; absent in the sandbox
    def __init__(self):
        self.v = 1

HAVE_RUNTIME = shutil.which("deno") or shutil.which("node")


def _echo_handler(kind, payload):
    if kind == "query":
        return f"ANS[{payload['task']}|{payload.get('data')}]"
    return [f"ANS[{j['task']}|{j.get('data')}]" for j in payload]


def _sandbox(**kw):
    sb = PyodideSandbox(allow_unsafe_node=True, **kw)
    sb.start()
    return sb


def test_sandbox_executes_in_wasm_and_isolates_filesystem():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = _sandbox()
    try:
        sb.set_context("hello world")
        r = sb.run("import os\nprint(sorted(os.listdir('/')))", _echo_handler)
        # WASM MEMFS root, never the host's real filesystem
        assert "home" in r.stdout and "usr" not in r.stdout
        assert "/mnt" not in r.stdout
    finally:
        sb.close()


def test_sandbox_blocks_js_escape():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = _sandbox()
    try:
        sb.set_context("x")
        r = sb.run("import js\nprint('ESCAPED', js)", _echo_handler)
        assert "ESCAPED" not in r.stdout
        assert "Error" in r.stdout or "None in sys.modules" in r.stdout
    finally:
        sb.close()


def test_sandbox_context_is_a_variable():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = _sandbox()
    try:
        sb.set_context("the magic number is 8675309")
        r = sb.run("print(len(context)); print(context[-7:])", _echo_handler)
        assert "8675309" in r.stdout
    finally:
        sb.close()


def test_sandbox_async_llm_query_bridges_to_host():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = _sandbox()
    try:
        sb.set_context("payload-data")
        r = sb.run("ans = await llm_query('extract', context)\nprint(ans)", _echo_handler)
        assert "ANS[extract|payload-data]" in r.stdout
    finally:
        sb.close()


def test_sandbox_batch_parallel_preserves_order():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = _sandbox()
    try:
        sb.set_context("x")
        code = ("res = await batch_llm_query("
                "[{'task': 'q', 'data': c} for c in ['A', 'B', 'C']])\nprint(res)")
        r = sb.run(code, _echo_handler)
        assert r.stdout.index("ANS[q|A]") < r.stdout.index("ANS[q|B]") < r.stdout.index("ANS[q|C]")
    finally:
        sb.close()


def test_sandbox_final_roundtrips_dict():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = _sandbox()
    try:
        sb.set_context("x")
        r = sb.run("FINAL({'answer': 42, 'ok': True})", _echo_handler)
        assert r.has_final and r.final == {"answer": 42, "ok": True}
    finally:
        sb.close()


def test_sandbox_execution_timeout_and_restart():
    if not HAVE_RUNTIME:
        print("SKIP"); return
    sb = _sandbox(exec_timeout=3)
    try:
        sb.set_context("ctx")
        assert "ExecutionTimeout" in sb.run("while True:\n    pass", _echo_handler).stdout
        assert "alive 2" in sb.run("print('alive', 1 + 1)", _echo_handler).stdout
    finally:
        sb.close()


def test_sandbox_namespace_reset_between_contexts():
    if not HAVE_RUNTIME:
        print("SKIP"); return
    sb = _sandbox()
    try:
        sb.set_context("A"); sb.run("leaked = 999", _echo_handler); sb.set_context("B")
        assert "False B" in sb.run("print('leaked' in globals(), context)", _echo_handler).stdout
    finally:
        sb.close()


def test_node_runtime_is_gated():
    import shutil as _sh
    if not _sh.which("node"):
        print("SKIP"); return
    try:
        PyodideSandbox(runtime="node"); assert False
    except RuntimeError as e:
        assert "allow_unsafe_node" in str(e)
    PyodideSandbox(runtime="node", allow_unsafe_node=True)


def test_depth2_recursion_isolates_parent_context():
    if not HAVE_RUNTIME:
        print("SKIP"); return
    parent = ["```repl\nans = await rlm_query('child task', data='x')\n```",
              "```repl\nFINAL(f'{ans}|parent_ctx={context}')\n```"]
    rlm = RLM(client=MockClient(responses=parent),
              sub_client=MockClient(responder=lambda m: "```repl\nFINAL('CHILD_OK')\n```"),
              sandbox=_sandbox(), max_iterations=4, max_depth=2)
    try:
        r = rlm.complete("outer", "PARENT_CTX")
        assert "CHILD_OK" in str(r.answer) and "parent_ctx=PARENT_CTX" in str(r.answer)
    finally:
        rlm.sandbox.close()


def test_sandbox_streams_large_context():
    """A context above the inline threshold is chunked and reassembled intact."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = _sandbox()
    try:
        big = ("A" * 1_500_000) + "NEEDLE_7777777" + ("B" * 1_500_000)
        sb.set_context(big)
        r = sb.run("print(len(context), 'NEEDLE_7777777' in context)", _echo_handler)
        assert str(len(big)) in r.stdout and "True" in r.stdout, r.stdout
        # large JSON context round-trips as an object
        sb.set_context({"items": [{"i": i} for i in range(60000)]})
        r = sb.run("print(len(context['items']), context['items'][-1])", _echo_handler)
        assert "60000" in r.stdout and "59999" in r.stdout, r.stdout
    finally:
        sb.close()


def test_sandbox_host_tool_channel():
    """Model calls `await tool(name, ...)`; the host-registered callable runs."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    def greet(name):
        "greet someone"
        return f"hi {name}"
    root = ["```repl\nFINAL(await tool('greet', name='bob'))\n```"]
    rlm = RLM(client=MockClient(responses=root), sub_client=MockClient(),
              sandbox=_sandbox(), tools={"greet": greet}, max_iterations=4)
    try:
        assert rlm.complete("q", "ctx").answer == "hi bob"
    finally:
        rlm.sandbox.close()


def test_sandbox_pickle_arbitrary_context():
    """Non-JSON objects cross via the pickle codec and reconstruct in the sandbox."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    from datetime import datetime
    from decimal import Decimal
    sb = _sandbox()
    try:
        sb.set_context({"when": datetime(2026, 6, 17), "amount": Decimal("10.50"),
                        "tags": {1, 2, 3}})
        r = sb.run("FINAL([context['when'].year, str(context['amount']), "
                   "sorted(context['tags'])])", _echo_handler)
        assert r.final == [2026, "10.50", [1, 2, 3]], r.final
    finally:
        sb.close()


def test_sandbox_pickle_unavailable_class_errors_clearly():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    sb = PyodideSandbox(allow_unsafe_node=True, context_codec="pickle"); sb.start()
    try:
        sb.set_context(_Unsendable())
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "could not be bound" in str(e)
    finally:
        sb.close()


def test_sandbox_pool_warm_reuse_and_replace():
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    pool = SandboxPool(factory=lambda: PyodideSandbox(allow_unsafe_node=True), size=1)
    try:
        a = pool.acquire(); pid1 = a._proc.pid; pool.release(a)
        b = pool.acquire()
        assert b._proc.pid == pid1                 # reused warm subprocess
        b._proc.kill(); b._proc.wait(); pool.release(b)
        c = pool.acquire()
        assert pool._alive(c) and c._proc.pid != pid1   # dead one replaced
        c.set_context("x")
        assert "OK2" in c.run("print('OK2')", _echo_handler).stdout
        pool.release(c)
    finally:
        pool.close()


def test_trace_captures_nested_run():
    """A full run records a span tree (root > root_turn/repl_cell > sub_query)."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    seen = []
    script = ["```repl\nans = await llm_query('label', data='x')\n```",
              "```repl\nFINAL(ans)\n```"]
    rlm = RLM(client=MockClient(responses=script),
              sub_client=MockClient(responder=lambda m: "LABELED"),
              sandbox=_sandbox(), max_iterations=4,
              on_span=lambda sp: seen.append(sp.kind))
    try:
        res = rlm.complete("q", "CTX")
        assert res.answer == "LABELED"
        kinds = [s["kind"] for s in res.trace]
        assert "root" in kinds and kinds.count("root_turn") >= 2
        assert "repl_cell" in kinds and "sub_query" in kinds
        byid = {s["span_id"]: s for s in res.trace}
        sub = [s for s in res.trace if s["kind"] == "sub_query"][0]
        assert byid[sub["parent_id"]]["kind"] == "repl_cell"   # sub-call nests under its cell
        assert seen, "on_span callback should fire live"
    finally:
        rlm.sandbox.close()


def test_llm_query_stays_leaf_at_depth2():
    """At max_depth=2, llm_query must NOT recurse (no child `rlm` span)."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    script = ["```repl\nans = await llm_query('x', data='y')\n```",
              "```repl\nFINAL(ans)\n```"]
    rlm = RLM(client=MockClient(responses=script),
              sub_client=MockClient(responder=lambda m: "LEAF"),
              sandbox=_sandbox(), max_iterations=4, max_depth=2)
    try:
        res = rlm.complete("q", "CTX")
        assert res.answer == "LEAF"
        kinds = [s["kind"] for s in res.trace]
        assert "sub_query" in kinds and "rlm" not in kinds   # leaf only, no recursion
    finally:
        rlm.sandbox.close()


def test_batch_rlm_query_recurses_per_item():
    """batch_rlm_query spawns a recursive child per job (each its own sandbox)."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    script = ["```repl\nres = await batch_rlm_query("
              "[{'task': 't', 'data': 'a'}, {'task': 't', 'data': 'b'}])\nFINAL(res)\n```"]
    rlm = RLM(client=MockClient(responses=script),
              sub_client=MockClient(responder=lambda m: "```repl\nFINAL('C')\n```"),
              sandbox=_sandbox(), max_iterations=4, max_depth=2)
    try:
        res = rlm.complete("q", "CTX")
        assert list(res.answer) == ["C", "C"], res.answer
        rlm_spans = [s for s in res.trace if s["kind"] == "rlm"]
        assert len(rlm_spans) == 2   # one recursive child per job
    finally:
        rlm.sandbox.close()


def test_trace_autosaves_jsonl_when_content_enabled():
    """complete() with trace_content writes <trace_dir>/<trace_id>.jsonl; reload == trace."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    import tempfile, os
    from rlm import load_jsonl
    script = ["```repl\nFINAL('ok')\n```"]
    with tempfile.TemporaryDirectory() as d:
        rlm = RLM(client=MockClient(responses=script), sub_client=MockClient(),
                  sandbox=_sandbox(), max_iterations=3,
                  trace_content=True, trace_dir=d)
        try:
            res = rlm.complete("q", "CTX")
            path = os.path.join(d, f"{res.trace_id}.jsonl")
            assert os.path.exists(path), "trace file not written"
            assert load_jsonl(path) == res.trace
        finally:
            rlm.sandbox.close()
    # disabled when trace_dir=None
    with tempfile.TemporaryDirectory() as d:
        rlm = RLM(client=MockClient(responses=["```repl\nFINAL('ok')\n```"]),
                  sub_client=MockClient(), sandbox=_sandbox(), max_iterations=3,
                  trace_content=True, trace_dir=None)
        try:
            rlm.complete("q", "CTX")
            assert os.listdir(d) == []   # nothing written
        finally:
            rlm.sandbox.close()


def test_sandbox_does_not_expose_provider_keys():
    """Defense in depth: a provider key in the host env is not visible inside the sandbox."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-should-not-leak"
    sb = _sandbox()
    try:
        sb.set_context("x")
        r = sb.run("import os\nprint(os.environ.get('ANTHROPIC_API_KEY'))", _echo_handler)
        assert "sk-should-not-leak" not in r.stdout
    finally:
        sb.close()


def test_full_rlm_through_real_sandbox():
    """End-to-end: scripted root model drives the real WASM sandbox to the answer."""
    if not HAVE_RUNTIME:
        print("SKIP (no deno/node)"); return
    needle = "8675309"
    context = "\n".join([f"line {i}: filler" for i in range(2000)])
    context = context.replace("line 1000: filler", f"line 1000: magic number {needle}")

    script = [
        "```repl\nprint(context[:60]); print(len(context))\n```",
        ("```repl\n"
         "hit = [l for l in context.split(chr(10)) if 'magic number' in l][0]\n"
         "ans = await llm_query('Return only the digits.', hit)\n"
         "print(ans)\n```"),
        "```repl\nFINAL(ans.strip())\n```",
    ]

    def sub(msgs):
        import re
        m = re.search(r"\d{7}", msgs[-1]["content"])
        return m.group(0) if m else "?"

    rlm = RLM(client=MockClient(responses=script),
              sub_client=MockClient(responder=sub),
              sandbox=_sandbox(), max_iterations=6)
    try:
        result = rlm.complete("What is the magic number?", context)
        assert result.answer == needle, f"got {result.answer!r}"
    finally:
        rlm.sandbox.close()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa
                import traceback
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print("\n" + ("ALL TESTS PASSED" if not failures else f"{failures} FAILURE(S)"))
    sys.exit(1 if failures else 0)
