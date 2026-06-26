"""
Command-line entry points.

  rlm [params] "query" context.txt        run a query over a context file
  rlm-tui trace.jsonl                      view a saved trace in the TUI
  rlm-tui --live [params] "query" file     run a query live in the TUI

Both are registered as console scripts (see pyproject `[project.scripts]`); you can
also use `python -m rlm.cli ...` / `python -m rlm.tui ...`. The run commands build a
provider-agnostic LiteLLM client, so install the `litellm` extra (and `tui` for the
viewer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .core import RLM
from .tracing import load_jsonl
from .tui import live_tui, run_tui


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", default=os.getenv("RLM_MODEL"),
                   help="root model (LiteLLM id), or $RLM_MODEL")
    p.add_argument("--sub-model", default=None, help="sub-call model (default: --model)")
    p.add_argument("--api-base", default=os.getenv("RLM_API_BASE"))
    p.add_argument("--api-key", default=os.getenv("RLM_API_KEY"))
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--sub-max-parallel", type=int, default=8)
    p.add_argument("--runtime", default="auto", choices=["auto", "deno", "node"])
    p.add_argument("--allow-unsafe-node", action="store_true")
    p.add_argument("--trace-content", action="store_true",
                   help="capture prompt/response text in the trace (and autosave it)")
    p.add_argument("--trace-dir", default="rlm_traces", help="autosave dir (or 'none')")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="parse the context file as JSON instead of text")


def _read_context(path: str, as_json: bool):
    if path == "-":
        data = sys.stdin.read()
    else:
        try:
            with open(path, encoding="utf-8") as f:
                data = f.read()
        except OSError as e:
            sys.exit(f"rlm: cannot read context {path!r}: {e}")
    return json.loads(data) if as_json else data


def _build_rlm(args, on_span=None) -> RLM:
    from .clients import LiteLLMClient
    from .sandbox import PyodideSandbox
    if not args.model:
        sys.exit("rlm: no model given (use --model or set RLM_MODEL)")
    client = LiteLLMClient(args.model, api_base=args.api_base, api_key=args.api_key)
    sub = (LiteLLMClient(args.sub_model, api_base=args.api_base, api_key=args.api_key)
           if args.sub_model else client)
    sandbox = PyodideSandbox(runtime=args.runtime, allow_unsafe_node=args.allow_unsafe_node)
    trace_dir = None if str(args.trace_dir).lower() == "none" else args.trace_dir
    return RLM(client=client, sub_client=sub, sandbox=sandbox,
               max_depth=args.max_depth, max_iterations=args.max_iterations,
               sub_max_parallel=args.sub_max_parallel,
               trace_content=args.trace_content, trace_dir=trace_dir, on_span=on_span)


def build_run_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlm", description="Run a Recursive Language Model query over a context file.")
    _add_model_args(p)
    p.add_argument("--show-usage", action="store_true", help="print token/cost usage to stderr")
    p.add_argument("query", help="the question/instruction")
    p.add_argument("context_file", help="path to the context file ('-' for stdin)")
    return p


def main(argv=None) -> int:
    args = build_run_parser().parse_args(argv)
    rlm = _build_rlm(args)
    res = rlm.complete(args.query, _read_context(args.context_file, args.as_json))
    print(res.answer)
    if args.show_usage:
        sys.stderr.write(f"[usage] {res.usage}\n")
    if res.trace_id and rlm.trace_content and rlm.trace_dir:
        sys.stderr.write(f"[trace] {rlm.trace_dir}/{res.trace_id}.jsonl\n")
    return 0


def build_tui_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlm-tui", description="View an RLM trace (saved JSONL), or run a query live in a TUI.")
    p.add_argument("--live", action="store_true", help="run a query live instead of viewing a file")
    _add_model_args(p)
    p.add_argument("args", nargs="*",
                   help="a trace .jsonl to view, or (with --live) QUERY CONTEXT_FILE")
    return p


def tui_main(argv=None) -> int:
    p = build_tui_parser()
    a = p.parse_args(argv)
    if a.live:
        if len(a.args) != 2:
            p.error("--live requires QUERY and CONTEXT_FILE")
        query, ctx_file = a.args
        rlm = _build_rlm(a)
        live_tui(rlm, query, _read_context(ctx_file, a.as_json))
    else:
        if len(a.args) != 1:
            p.error("provide a trace .jsonl path to view (or use --live to run one)")
        run_tui(load_jsonl(a.args[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
