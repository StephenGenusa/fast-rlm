# Observability (LLM tracing)

Every model interaction in a run is recorded as a structured **span**, so you can
see exactly what the RLM did: each root turn, REPL cell, leaf sub-call, recursive
child run, tool call, and schema-repair call — with timing, tokens, cost, depth,
and a parent link. Spans form a tree under one `trace_id`.

## Getting the trace

```python
res = rlm.complete(query, context)
for s in res.trace:           # list of span dicts
    print(s["kind"], s["model"], s["latency_s"], s["cost_usd"])
```

`RLMResult.trace` replaces the old flat `trajectory` (a `repl_cell` span carries
the same `code`/`stdout`, plus timing and its place in the tree).

## Live streaming (for a TUI / logger)

Pass `on_span` to get each span the moment it completes (fires from worker threads
for parallel sub-calls; the callback runs outside the tracer lock, so it may be invoked concurrently and must be thread-safe):

```python
rlm = RLM(client=..., on_span=lambda sp: print(sp.kind, sp.name, sp.latency_s))
```

## Span shape

`Span`: `trace_id, span_id, parent_id, kind, name, depth, model, start, end,
latency_s, input_tokens, output_tokens, cost_usd, content, error, metadata`.

`kind` is one of: `root` / `rlm` (a run), `root_turn` (root model call),
`repl_cell` (a code cell), `sub_query` / `sub_batch_item` (leaf sub-calls),
`tool` (host tool), `schema_repair`. Sub-call and tool spans nest under the
`repl_cell` that triggered them; a recursive child's spans nest under that cell too.

## Content capture / privacy

Prompt/response/code/stdout/tool-I/O text is **off by default** (context may hold
secrets); only timing, tokens, cost, depth, and the tree are always recorded. Turn
text on explicitly, and optionally scrub it:

```python
RLM(client=..., trace_content=True)                       # capture text (clipped to 2000 chars)
RLM(client=..., trace_content=True, redact=my_scrubber)   # capture, then redact each string
```

## Exporting

`Tracer.to_list()` / `to_jsonl()` dump the spans; `Span.to_otel()` maps a span to
an OpenTelemetry-GenAI-style dict (`gen_ai.*` attributes, trace/span ids), so
wiring an OTel or Langfuse exporter is a thin layer on top, not a rewrite.

## Saving traces (autosave)

When `trace_content=True`, each run autosaves its trace to
`<trace_dir>/<trace_id>.jsonl` (default `trace_dir="rlm_traces"`); `RLMResult.trace_id`
tells you which file. Set `trace_dir=None` to disable. Autosave is best-effort — a
write failure warns but never breaks the run. Note the file contains whatever was
captured, so a content-on trace may hold prompt/response text on disk (the `redact`
hook applies before saving). Manual save: `Tracer(...).save(path)` /
`rlm.tracing.load_jsonl(path)`.

## Viewing a trace (Textual TUI)

Install the extra (`pip install -e ".[tui]"`), then browse a trace in a master/detail
terminal UI — a navigable span list on the left, full span detail on the right,
run summary in the header:

```python
from rlm import run_tui, load_jsonl
run_tui(load_jsonl("rlm_traces/<trace_id>.jsonl"))   # a past run
```

or from the shell: `python -m rlm.tui rlm_traces/<trace_id>.jsonl`.

**Live mode** runs a query and fills the UI in real time as spans arrive:

```python
from rlm import RLM, run_tui, live_tui
rlm = RLM(client=..., trace_content=True)   # content on so the detail panel is useful
live_tui(rlm, query="…", context=huge_text)
```

(`live_tui` runs `complete()` in a daemon thread and pushes spans to the UI via the
`on_span` callback; `q` quits.)
