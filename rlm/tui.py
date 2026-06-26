"""
Textual trace viewer for RLM runs (master/detail).

Two entry points:
  - run_tui(spans)                 — browse a completed trace (list of span dicts)
  - live_tui(rlm, query, context)  — run and watch the trace fill in live

Or from the shell:  python -m rlm.tui path/to/trace.jsonl

`textual` is an optional dependency (`pip install -e ".[tui]"`). The pure
formatters below have no textual/rich dependency and are unit-tested directly.
"""

from __future__ import annotations

from typing import Any

# Columns shown in the left list.
_COLUMNS = ("event", "name", "model", "lat(s)", "in", "out", "$")
_GLYPH = {
    "root": "●", "rlm": "◆", "root_turn": "▸", "repl_cell": "⌨",
    "sub_query": "↳", "sub_batch_item": "↳", "tool": "⚙", "schema_repair": "✎",
}


# --- pure formatters (no textual/rich) --------------------------------------
def _row_cells(span: dict) -> tuple[str, ...]:
    depth = int(span.get("depth", 0) or 0)
    kind = span.get("kind", "")
    name = ("  " * depth) + (span.get("name") or kind)
    return (
        f"{_GLYPH.get(kind, '·')} {kind}",
        name,
        span.get("model") or "",
        f"{float(span.get('latency_s', 0) or 0):.2f}",
        str(span.get("input_tokens", 0) or 0),
        str(span.get("output_tokens", 0) or 0),
        f"${float(span.get('cost_usd', 0) or 0):.4f}",
    )


def _detail_text(span: dict) -> str:
    g = span.get
    lines = [
        f"{g('kind', '')}    {g('name', '')}",
        "",
        f"span_id   : {g('span_id', '')}",
        f"parent_id : {g('parent_id', '')}",
        f"trace_id  : {g('trace_id', '')}",
        f"depth     : {g('depth', 0)}    model: {g('model') or '—'}",
        f"latency   : {float(g('latency_s', 0) or 0):.3f}s    "
        f"tokens: in {g('input_tokens', 0) or 0} / out {g('output_tokens', 0) or 0}    "
        f"cost: ${float(g('cost_usd', 0) or 0):.4f}",
    ]
    if g("error"):
        lines += ["", f"ERROR: {g('error')}"]
    content = g("content") or {}
    if content:
        lines += ["", "── content ──"]
        for k, v in content.items():
            lines += [f"[{k}]", str(v), ""]
    else:
        lines += ["", "(content not captured — set trace_content=True)"]
    md = g("metadata") or {}
    if md:
        lines += ["── metadata ──", str(md)]
    return "\n".join(lines)


def _totals(spans: list[dict]) -> dict:
    roots = [s for s in spans if s.get("kind") == "root"]
    if roots:
        intok = sum(s.get("input_tokens", 0) or 0 for s in roots)
        outtok = sum(s.get("output_tokens", 0) or 0 for s in roots)
        cost = sum(float(s.get("cost_usd", 0) or 0) for s in roots)
    else:
        model_kinds = {"root_turn", "sub_query", "sub_batch_item", "schema_repair"}
        msp = [s for s in spans if s.get("kind") in model_kinds]
        intok = sum(s.get("input_tokens", 0) or 0 for s in msp)
        outtok = sum(s.get("output_tokens", 0) or 0 for s in msp)
        cost = sum(float(s.get("cost_usd", 0) or 0) for s in msp)
    starts = [s.get("start", 0) for s in spans if s.get("start")]
    ends = [s.get("end", 0) for s in spans if s.get("end")]
    wall = (max(ends) - min(starts)) if starts and ends else 0.0
    tid = (spans[0].get("trace_id") if spans else "") or ""
    return {"spans": len(spans), "in": intok, "out": outtok, "cost": cost,
            "wall": wall, "trace_id": tid}


def _summary(spans: list[dict]) -> str:
    t = _totals(spans)
    return (f"{t['spans']} spans · in {t['in']} / out {t['out']} tok · "
            f"${t['cost']:.4f} · {t['wall']:.2f}s · trace {t['trace_id'][:8]}")


# --- Textual app ------------------------------------------------------------
def _build_app():
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import DataTable, Footer, Header, Static

    class TraceApp(App):
        CSS = """
        #list { width: 55%; border-right: solid $accent; }
        #detailwrap { width: 45%; padding: 0 1; }
        #detail { width: 100%; }
        """
        BINDINGS = [("q", "quit", "Quit"), ("g", "top", "Top"), ("G", "bottom", "Bottom")]

        def __init__(self, spans: list[dict] | None = None, live: bool = False,
                     on_ready=None):
            super().__init__()
            self.spans: list[dict] = []
            self._initial = list(spans or [])
            self.live = live
            self._on_ready = on_ready
            self._status = "live…" if live else "loaded"
            self.detail_text = ""

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Horizontal():
                yield DataTable(id="list", cursor_type="row", zebra_stripes=True)
                with VerticalScroll(id="detailwrap"):
                    yield Static("", id="detail")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#list", DataTable)
            table.add_columns(*_COLUMNS)
            self.title = "RLM trace"
            self._refresh_summary()
            for sp in self._initial:
                self._append(sp)
            if self.spans:
                table.move_cursor(row=0)
                self._show(0)
            if self._on_ready is not None:
                self._on_ready()

        # -- span ingestion (static + live) --------------------------------
        def add_span(self, span) -> None:
            sp = span.to_dict() if hasattr(span, "to_dict") else dict(span)
            self._append(sp)
            table = self.query_one("#list", DataTable)
            if self.live:
                table.move_cursor(row=len(self.spans) - 1)  # follow tail
                self._show(len(self.spans) - 1)
            self._refresh_summary()

        def _append(self, sp: dict) -> None:
            self.spans.append(sp)
            self.query_one("#list", DataTable).add_row(*_row_cells(sp))

        def mark_done(self, error=None) -> None:
            self._status = f"error: {error}" if error else "done"
            self._refresh_summary()

        # -- master -> detail ----------------------------------------------
        def on_data_table_row_highlighted(self, event) -> None:
            self._show(event.cursor_row)

        def _show(self, row: int) -> None:
            if 0 <= row < len(self.spans):
                self.detail_text = _detail_text(self.spans[row])
                self.query_one("#detail", Static).update(self.detail_text)

        def _refresh_summary(self) -> None:
            base = _summary(self.spans) if self.spans else "no spans yet"
            self.sub_title = f"{base}  [{self._status}]"

        # -- actions --------------------------------------------------------
        def action_top(self) -> None:
            if self.spans:
                self.query_one("#list", DataTable).move_cursor(row=0)

        def action_bottom(self) -> None:
            if self.spans:
                self.query_one("#list", DataTable).move_cursor(row=len(self.spans) - 1)

    return TraceApp


def run_tui(spans: list[dict]):
    """Open the viewer on a completed trace (list of span dicts)."""
    return _build_app()(spans=spans).run()


def live_tui(rlm, query, context, **complete_kwargs):
    """Run `rlm.complete(query, context)` in a worker thread and watch live."""
    import threading

    app_cls = _build_app()
    app = app_cls(live=True)

    def _runner():
        prev = rlm.on_span
        rlm.on_span = lambda sp: app.call_from_thread(app.add_span, sp)
        try:
            rlm.complete(query, context, **complete_kwargs)
            app.call_from_thread(app.mark_done, None)
        except Exception as e:  # noqa - surface in the footer
            app.call_from_thread(app.mark_done, e)
        finally:
            rlm.on_span = prev

    app._on_ready = lambda: threading.Thread(target=_runner, daemon=True).start()
    return app.run()


if __name__ == "__main__":
    import sys
    from .tracing import load_jsonl
    if len(sys.argv) != 2:
        print("usage: python -m rlm.tui <trace.jsonl>"); sys.exit(2)
    run_tui(load_jsonl(sys.argv[1]))
