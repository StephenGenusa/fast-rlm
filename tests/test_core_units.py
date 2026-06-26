"""core.py: pure helpers + complete() control-flow branches (offline, scripted sandbox)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _fakes

from rlm import RLM, MockClient, SandboxPool                                  # noqa: E402
from rlm.core import _context_meta, _as_text, _coerce, _validate             # noqa: E402
from _fakes import ScriptedSandbox, sresult                                   # noqa: E402


# -- helpers ----------------------------------------------------------------
def test_context_meta_by_type():
    assert "type=str" in _context_meta("hello")
    assert "top-level keys" in _context_meta({"a": 1})
    assert "type=list" in _context_meta([1, 2, 3])
    assert "type=int" in _context_meta(123)


def test_as_text_variants():
    assert _as_text("raw") == "raw"                       # str passthrough (uncapped)
    assert _as_text({"a": 1}) == '{"a": 1}'               # json
    assert _as_text(object()).startswith("<object")       # non-serializable -> str()
    assert len(_as_text(["x"] * 200000)) <= 500_000       # cap on non-str


def test_coerce_paths():
    assert _coerce("x", None) == "x"                      # no schema
    assert _coerce('{"a":1}', {}) == {"a": 1}             # plain json
    assert _coerce("```json\n{\"b\":2}\n```", {}) == {"b": 2}   # fenced json
    assert _coerce("not json", {}) == "not json"          # falls back to raw


def test_validate_paths():
    schema = {"type": "object", "required": ["v"], "properties": {"v": {"type": "integer"}}}
    assert _validate(None, None) == (True, None)          # no schema
    try:
        import jsonschema  # noqa
        assert _validate({"v": 1}, schema)[0] is True
        assert _validate({"v": "x"}, schema)[0] is False
        assert _validate({}, schema)[0] is False
    except ImportError:
        assert _validate({"v": "x"}, schema) == (True, None)   # parse-only fallback


# -- complete() branches via a scripted sandbox -----------------------------
def _rlm(responses, results, **kw):
    return RLM(client=MockClient(responses=responses),
               sandbox=ScriptedSandbox(results=results), max_iterations=kw.pop("mi", 6), **kw)


def test_complete_final_via_sandbox():
    r = _rlm(["```repl\nFINAL('ANS')\n```"], [sresult("out", "ANS", True)])
    res = r.complete("q", "ctx")
    assert res.answer == "ANS" and res.iterations == 1
    kinds = [s["kind"] for s in res.trace]
    assert "root" in kinds and "root_turn" in kinds and "repl_cell" in kinds


def test_complete_text_final_fallback_no_block():
    r = _rlm(["the answer is FINAL(prose)"], [])
    assert r.complete("q", "ctx").answer == "prose"      # text-FINAL, no repl block


def test_complete_nudge_then_progress():
    r = _rlm(["no code here at all", "```repl\nFINAL('later')\n```"],
             [sresult("o", "later", True)])
    res = r.complete("q", "ctx")
    assert res.answer == "later" and res.iterations == 2  # iter0 nudged, iter1 finals


def test_complete_forced_final_at_iteration_cap():
    def responder(msgs):
        last = msgs[-1]["content"]
        return "FINAL(capped)" if "out of iterations" in last else "```repl\nprint(1)\n```"
    r = RLM(client=MockClient(responder=responder),
            sandbox=ScriptedSandbox(on_run=lambda c, h: sresult("x", None, False)),
            max_iterations=2)
    assert r.complete("q", "ctx").answer == "capped"


def test_complete_via_pool_reuses_sandbox():
    pool = SandboxPool(factory=lambda: ScriptedSandbox(results=[sresult("o", "P", True)]), size=1)
    r = RLM(client=MockClient(responder=lambda m: "```repl\nFINAL('P')\n```"), sandbox_pool=pool)
    assert r.complete("q", "a").answer == "P"
    sb = pool.acquire()
    assert sb.reset_calls >= 1 and sb.started   # released + reset back into the pool
    pool.close()


def test_unknown_sub_kind_raises():
    r = RLM(client=MockClient())
    try:
        r._make_sub_handler(0)("bogus", {})
        assert False
    except ValueError:
        pass


def test_tool_exception_is_surfaced():
    def boom():
        raise RuntimeError("kaboom")
    r = RLM(client=MockClient(), tools={"boom": boom})
    out = r._make_sub_handler(0)("tool", {"name": "boom", "args": [], "kwargs": {}})
    assert out.startswith("[tool error: RuntimeError")


def test_validate_handles_bad_schema():
    ok, err = _validate({"x": 1}, {"type": "not-a-real-type"})
    try:
        import jsonschema  # noqa
        assert ok is False and err   # malformed schema -> handled, not raised
    except ImportError:
        assert ok is True


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa
                failures += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\n" + ("ALL TESTS PASSED" if not failures else f"{failures} FAILURE(S)"))
    sys.exit(1 if failures else 0)
