"""TUI / trace-persistence tests. Textual-dependent tests skip if textual absent."""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.tui import _row_cells, _detail_text, _summary, _totals   # noqa: E402
from rlm.tracing import Tracer, load_jsonl                        # noqa: E402

SAMPLE = [
    {"trace_id": "t", "span_id": "a", "parent_id": None, "kind": "root",
     "name": "rlm(depth=0)", "depth": 0, "model": "m", "latency_s": 1.5,
     "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.01, "content": {},
     "error": None, "metadata": {}, "start": 1.0, "end": 2.5},
    {"trace_id": "t", "span_id": "b", "parent_id": "a", "kind": "sub_query",
     "name": "llm_query", "depth": 1, "model": "mini", "latency_s": 0.4,
     "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001,
     "content": {"prompt": "label this", "response": "DONE"},
     "error": None, "metadata": {}, "start": 1.2, "end": 1.6},
    {"trace_id": "t", "span_id": "c", "parent_id": "a", "kind": "tool",
     "name": "tool:add", "depth": 1, "model": None, "latency_s": 0.0,
     "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "content": {},
     "error": "[tool error: boom]", "metadata": {}, "start": 1.7, "end": 1.7},
]


def test_row_cells_indents_by_depth_and_formats():
    root = _row_cells(SAMPLE[0])
    sub = _row_cells(SAMPLE[1])
    assert root[0].endswith("root") and "$0.0100" in root
    assert sub[1].startswith("  ") and sub[1].strip() == "llm_query"   # depth indent
    assert root[3] == "1.50"   # latency


def test_detail_text_shows_content_error_and_empty():
    d = _detail_text(SAMPLE[1])
    assert "[prompt]" in d and "label this" in d and "[response]" in d
    assert "(content not captured" in _detail_text(SAMPLE[0])      # empty content
    assert "ERROR: [tool error: boom]" in _detail_text(SAMPLE[2])  # error span


def test_totals_and_summary():
    t = _totals(SAMPLE)
    assert t["spans"] == 3 and t["in"] == 100 and t["out"] == 20   # from the root span
    assert abs(t["cost"] - 0.01) < 1e-9 and abs(t["wall"] - 1.5) < 1e-6
    assert "3 spans" in _summary(SAMPLE)


def test_save_and_load_jsonl_roundtrip():
    t = Tracer()
    with t.span("root_turn", "turn 0", model="m") as sp:
        sp.input_tokens = 7
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "trace.jsonl"          # nested dir is created
        t.save(p)
        assert load_jsonl(p) == t.to_list()


def test_app_master_detail_headless():
    try:
        import textual  # noqa
    except ImportError:
        print("SKIP (textual not installed)"); return
    from rlm.tui import _build_app

    async def _run():
        app = _build_app()(spans=SAMPLE)
        async with app.run_test() as pilot:
            table = app.query_one("#list")
            assert table.row_count == len(SAMPLE)
            await pilot.press("down")                # move highlight to row 1
            assert table.cursor_row == 1
            assert "llm_query" in app.detail_text and "label this" in app.detail_text
    asyncio.run(_run())


def test_app_live_add_span_headless():
    try:
        import textual  # noqa
    except ImportError:
        print("SKIP (textual not installed)"); return
    from rlm.tui import _build_app

    async def _run():
        app = _build_app()(live=True)
        async with app.run_test() as pilot:
            assert app.query_one("#list").row_count == 0
            for sp in SAMPLE:
                app.add_span(sp)
            await pilot.pause()
            assert app.query_one("#list").row_count == len(SAMPLE)
            app.mark_done(None)
            assert "done" in app.sub_title
    asyncio.run(_run())


def test_totals_fallback_without_root_span():
    from rlm.tui import _totals
    spans = [s for s in SAMPLE if s["kind"] != "root"]   # only sub_query + tool
    t = _totals(spans)
    assert t["spans"] == 2 and t["in"] == 10 and t["out"] == 5   # summed model spans
    assert _totals([])["spans"] == 0                              # empty is safe


def test_row_cells_unknown_kind_uses_default_glyph():
    cells = _row_cells({"kind": "mystery", "name": "n", "depth": 0})
    assert cells[0].startswith("·") and "mystery" in cells[0]


def test_live_tui_wires_on_span_and_marks_done():
    import rlm.tui as tui
    from rlm.core import RLMResult
    from rlm.clients import Usage
    captured = {"spans": [], "done": None}

    class FakeApp:
        def __init__(self, live=False): self._on_ready = None
        def call_from_thread(self, fn, *a): fn(*a)
        def add_span(self, sp): captured["spans"].append(sp)
        def mark_done(self, e=None): captured["done"] = "err" if e else "ok"
        def run(self):
            if self._on_ready:
                self._on_ready()
            import time; time.sleep(0.2)

    class FakeRLM:
        on_span = None
        def complete(self, q, c, **k):
            for sp in [{"kind": "root_turn"}, {"kind": "sub_query"}]:
                self.on_span(sp)
            return RLMResult("A", Usage(), 1, [], "t")

    orig = tui._build_app
    tui._build_app = lambda: FakeApp
    try:
        tui.live_tui(FakeRLM(), "q", "ctx")
        assert len(captured["spans"]) == 2 and captured["done"] == "ok"
    finally:
        tui._build_app = orig


def test_app_nav_actions_top_bottom():
    try:
        import textual  # noqa
    except ImportError:
        print("SKIP (textual)"); return
    from rlm.tui import _build_app

    async def _run():
        app = _build_app()(spans=SAMPLE)
        async with app.run_test() as pilot:
            await pilot.press("G")
            assert app.query_one("#list").cursor_row == len(SAMPLE) - 1
            await pilot.press("g")
            assert app.query_one("#list").cursor_row == 0
    asyncio.run(_run())


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
