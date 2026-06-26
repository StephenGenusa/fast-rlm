"""prompts.py: message construction, the query/context separation invariant, step prompts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm.prompts import build_messages, step_prompt, _render_system   # noqa: E402
from rlm.core import _context_meta                                    # noqa: E402


def test_build_messages_roles_and_query_present():
    msgs = build_messages("MY_QUERY", "type=str, length=10 chars")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "MY_QUERY" in msgs[1]["content"]
    assert "type=str" in msgs[1]["content"]


def test_full_context_is_never_embedded():
    # the separation invariant: only metadata (a short preview) reaches the prompt,
    # never the full context — a marker past the 300-char preview must be absent.
    ctx = ("A" * 1000) + "SECRET_MARKER_XYZ"
    meta = _context_meta(ctx)
    msgs = build_messages("q", meta)
    blob = msgs[0]["content"] + msgs[1]["content"]
    assert "SECRET_MARKER_XYZ" not in blob


def test_step_prompt_branches():
    assert "Inspect" in step_prompt("q", 0)["content"]
    assert "Continue" in step_prompt("q", 3)["content"]
    assert "out of iterations" in step_prompt("q", 5, force_final=True)["content"]


def test_render_system_async_vs_sync():
    a = _render_system(async_subcalls=True)
    s = _render_system(async_subcalls=False)
    assert "asynchronous" in a and "await llm_query" in a
    assert "synchronous" in s and "do NOT write await" in s


def test_render_system_tools_and_recursion_conditional():
    base = _render_system(True)
    assert "rlm_query" not in base and "Host tools" not in base
    withtools = _render_system(True, tools_doc="- search: web search")
    assert "Host tools" in withtools and "search: web search" in withtools
    withrec = _render_system(True, recursive_enabled=True)
    assert "rlm_query" in withrec and "batch_rlm_query" in withrec


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
