"""
Structured per-call tracing for RLM runs.

Every model interaction in a run — the root turns, REPL cells, leaf sub-calls,
recursive child runs, tool calls, and schema-repair calls — is recorded as a
``Span`` with timing, tokens, cost, depth, and a parent link. Spans form a tree
under a single ``trace_id``; the same ``Tracer`` is shared across recursion so
child spans nest under the call that spawned them.

The schema is OpenTelemetry-GenAI-friendly (`Span.to_otel()`), so wiring an
exporter (OTel / Langfuse) or a TUI on top is mechanical. No external deps.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str          # root | rlm | root_turn | repl_cell | sub_query | sub_batch_item | tool | schema_repair
    name: str
    depth: int = 0
    model: str | None = None
    start: float = 0.0
    end: float = 0.0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    content: dict = field(default_factory=dict)   # e.g. {"prompt":..,"response":..} or {"code":..,"stdout":..}
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_otel(self) -> dict:
        """Map to an OpenTelemetry-GenAI-style span dict."""
        attrs = {
            "gen_ai.operation.name": self.kind,
            "gen_ai.request.model": self.model,
            "gen_ai.usage.input_tokens": self.input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
            "rlm.cost_usd": self.cost_usd,
            "rlm.depth": self.depth,
        }
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_id,
            "start_time_unix": self.start,
            "end_time_unix": self.end,
            "status": "ERROR" if self.error else "OK",
            "attributes": {k: v for k, v in attrs.items() if v is not None},
        }


class Tracer:
    """Thread-safe span recorder for one RLM run (shared across recursion)."""

    def __init__(self, on_span: Callable[[Span], None] | None = None,
                 trace_content: bool = False, content_max: int = 2000,
                 redact: Callable[[str], str] | None = None,
                 trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self._on_span = on_span
        self.trace_content = trace_content   # off by default — content may hold secrets
        self.content_max = content_max
        self._redact = redact                # optional scrubber applied to captured text
        self._lock = threading.Lock()
        self.spans: list[Span] = []

    # -- content capture ----------------------------------------------------
    def _clip(self, s: Any) -> str:
        s = s if isinstance(s, str) else str(s)
        if len(s) <= self.content_max:
            return s
        return s[: self.content_max] + f"… [{len(s) - self.content_max} chars clipped]"

    def content(self, **kw: Any) -> dict:
        if not self.trace_content:
            return {}
        out = {}
        for k, v in kw.items():
            if v is None:
                continue
            text = self._clip(v)
            if self._redact is not None:
                try:
                    text = self._redact(text)
                except Exception:  # a bad redactor must not break a run  # noqa
                    text = "[redaction error]"
            out[k] = text
        return out

    # -- span lifecycle -----------------------------------------------------
    def span(self, kind: str, name: str, parent_id: str | None = None,
             depth: int = 0, model: str | None = None) -> "_SpanCtx":
        return _SpanCtx(self, Span(
            trace_id=self.trace_id, span_id=uuid.uuid4().hex, parent_id=parent_id,
            kind=kind, name=name, depth=depth, model=model))

    def open(self, kind: str, name: str, parent_id: str | None = None,
             depth: int = 0, model: str | None = None) -> Span:
        """Create + start a span you will finish later with close() (for spans
        that wrap a whole loop, like the run-level root span)."""
        sp = Span(trace_id=self.trace_id, span_id=uuid.uuid4().hex, parent_id=parent_id,
                  kind=kind, name=name, depth=depth, model=model)
        sp.start = time.time()
        return sp

    def close(self, span: Span) -> None:
        span.end = time.time()
        if not span.latency_s:
            span.latency_s = round(span.end - span.start, 6)
        self._record(span)

    def _record(self, span: Span) -> None:
        # Append under the lock, but fire the callback OUTSIDE it: the callback is
        # foreign code that may call back into the tracer (to_list/to_jsonl), which
        # would deadlock a non-reentrant lock. It may therefore run concurrently
        # from parallel sub-call threads — on_span consumers must be thread-safe.
        with self._lock:
            self.spans.append(span)
            cb = self._on_span
        if cb is not None:
            try:
                cb(span)
            except Exception:  # a bad consumer must never break a run  # noqa
                pass

    # -- export -------------------------------------------------------------
    def to_list(self) -> list[dict]:
        with self._lock:
            return [s.to_dict() for s in self.spans]

    def to_jsonl(self) -> str:
        with self._lock:
            return "\n".join(json.dumps(s.to_dict()) for s in self.spans)

    def save(self, path) -> None:
        """Write the spans as JSON-lines to `path` (one span per line)."""
        import os
        path = os.fspath(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_jsonl())


def load_jsonl(path) -> list[dict]:
    """Load a saved trace (one JSON span per line) into a list of span dicts."""
    import os
    spans = []
    with open(os.fspath(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


class _SpanCtx:
    """Context manager that times a span and records it (with any error) on exit."""

    def __init__(self, tracer: Tracer, span: Span):
        self._tracer = tracer
        self.span = span

    def __enter__(self) -> Span:
        self.span.start = time.time()
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.span.end = time.time()
        if not self.span.latency_s:  # caller may set a more precise provider latency
            self.span.latency_s = round(self.span.end - self.span.start, 6)
        if exc is not None and not self.span.error:
            self.span.error = f"{exc_type.__name__}: {exc}"
        self._tracer._record(self.span)
        return False  # never suppress exceptions
