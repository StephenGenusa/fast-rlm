"""CLI tests — arg parsing, RLM construction, and TUI dispatch (no network/terminal)."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _fakes

import rlm.cli as cli  # noqa: E402


def _has_runtime():
    import shutil
    return bool(shutil.which("deno") or shutil.which("node"))


def test_run_parser_builds_rlm_and_reads_context():
    args = cli.build_run_parser().parse_args(
        ["-m", "openrouter/x", "--sub-model", "openrouter/y", "--max-depth", "2",
         "--allow-unsafe-node", "what?", "ctx.txt"])
    assert args.query == "what?" and args.context_file == "ctx.txt"
    if not _has_runtime():
        print("SKIP build (no runtime)"); return
    r = cli._build_rlm(args)
    assert r.max_depth == 2 and r.client.model == "openrouter/x" and r.sub_client.model == "openrouter/y"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.txt"); open(p, "w").write("hello")
        assert cli._read_context(p, False) == "hello"
        pj = os.path.join(d, "c.json"); open(pj, "w").write('{"a":1}')
        assert cli._read_context(pj, True) == {"a": 1}


def test_trace_dir_none_disables_autosave():
    if not _has_runtime():
        print("SKIP (no runtime)"); return
    args = cli.build_run_parser().parse_args(
        ["-m", "x", "--allow-unsafe-node", "--trace-dir", "none", "q", "f"])
    assert cli._build_rlm(args).trace_dir is None


def test_main_requires_a_model():
    os.environ.pop("RLM_MODEL", None)
    try:
        cli.main(["q", "f"])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "model" in str(e.code).lower()


def test_tui_view_dispatches_to_run_tui(monkeypatch=None):
    captured = {}
    orig = cli.run_tui
    cli.run_tui = lambda spans: captured.setdefault("spans", spans)
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.jsonl")
            open(p, "w").write(json.dumps({"kind": "root", "span_id": "a"}) + "\n")
            cli.tui_main([p])
        assert captured["spans"] == [{"kind": "root", "span_id": "a"}]
    finally:
        cli.run_tui = orig


def test_tui_live_dispatches_to_live_tui():
    captured = {}
    orig = cli.live_tui
    cli.live_tui = lambda r, q, c, **k: captured.update(q=q, c=c)
    try:
        if not _has_runtime():
            print("SKIP (no runtime)"); return
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "ctx.txt"); open(f, "w").write("CTX")
            cli.tui_main(["--live", "-m", "x", "--allow-unsafe-node", "myq", f])
        assert captured["q"] == "myq" and captured["c"] == "CTX"
    finally:
        cli.live_tui = orig


def test_tui_live_requires_two_positionals():
    try:
        cli.tui_main(["--live", "-m", "x", "onlyone"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_main_prints_answer_usage_and_trace_line():
    import io, contextlib, tempfile, os
    from _fakes import FakeRLM
    orig = cli._build_rlm
    cli._build_rlm = lambda args, on_span=None: FakeRLM(answer="HELLO", trace_content=True)
    try:
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "c.txt"); open(f, "w").write("ctx")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(["-m", "x", "--show-usage", "q", f])
            assert rc == 0 and out.getvalue().strip() == "HELLO"
            assert "[usage]" in err.getvalue() and "[trace]" in err.getvalue()
    finally:
        cli._build_rlm = orig


def test_read_context_from_stdin():
    import io
    orig = sys.stdin
    sys.stdin = io.StringIO("piped-context")
    try:
        assert cli._read_context("-", False) == "piped-context"
    finally:
        sys.stdin = orig


def test_read_context_missing_file_exits_cleanly():
    try:
        cli._read_context("/no/such/file_xyz_123", False); assert False
    except SystemExit as e:
        assert "cannot read context" in str(e.code)


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
