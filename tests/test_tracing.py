"""tracing.py edge cases: to_otel, to_jsonl, clipping, open/close, custom trace_id."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.tracing import Span, Tracer   # noqa: E402


def test_to_otel_status_and_none_omission():
    ok = Span("t", "s1", None, "root_turn", "turn", model=None).to_otel()
    assert ok["status"] == "OK" and "gen_ai.request.model" not in ok["attributes"]
    err = Span("t", "s2", None, "tool", "tool:x", error="boom").to_otel()
    assert err["status"] == "ERROR"


def test_to_jsonl_roundtrips_to_list():
    t = Tracer()
    with t.span("root_turn", "a"):
        pass
    with t.span("sub_query", "b"):
        pass
    lines = t.to_jsonl().splitlines()
    assert len(lines) == 2
    assert [json.loads(x) for x in lines] == t.to_list()


def test_content_clip_boundary():
    t = Tracer(trace_content=True, content_max=5)
    assert t.content(x="12345")["x"] == "12345"          # exactly at limit -> no clip
    clipped = t.content(x="123456")["x"]
    assert "clipped" in clipped and clipped.startswith("12345")


def test_open_close_sets_latency_and_records():
    t = Tracer()
    sp = t.open("root", "run")
    t.close(sp)
    assert len(t.spans) == 1 and t.spans[0].latency_s >= 0


def test_custom_trace_id_propagates():
    t = Tracer(trace_id="FIXED")
    with t.span("x", "x") as sp:
        pass
    assert t.trace_id == "FIXED" and t.spans[0].trace_id == "FIXED"


def test_to_otel_keeps_zero_valued_attrs():
    d = Span("t", "s", None, "root_turn", "n",
             input_tokens=0, output_tokens=0, cost_usd=0.0).to_otel()
    assert d["attributes"]["gen_ai.usage.input_tokens"] == 0
    assert d["attributes"]["rlm.cost_usd"] == 0.0


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
