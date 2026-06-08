"""Smoketest the benchmark harness WITHOUT spending money.

Validates scoring logic + dataset/example construction. It never calls
`fast_rlm.run`, so no LLM/engine is invoked. Run:

    uv run benchmarks/_smoketest.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _harness import exact_match, numeric_match, f1_score, best_of  # noqa: E402

PASS, FAIL = "✔", "✗"
_failures = []


def check(label, cond):
    print(f"  {PASS if cond else FAIL} {label}")
    if not cond:
        _failures.append(label)


def test_scorers():
    print("\n[scorers]")
    check("exact_match normalizes punctuation/case", exact_match("The Cat.", "the cat")[0])
    check("exact_match rejects mismatch", not exact_match("dog", "cat")[0])
    check("numeric_match exact", numeric_match(7, "7")[0])
    check("numeric_match pulls int from text", numeric_match("the answer is 42", 42)[0])
    check("numeric_match rejects wrong", not numeric_match(7, 8)[0])
    check("numeric_match tolerance", numeric_match(10.0, 10.4, tol=0.5)[0])
    check("f1 full overlap", f1_score("paris france", "paris france")[0])
    check("f1 partial below thresh fails",
          not f1_score("a b c d e", "x", threshold=0.5)[0])
    check("best_of matches one of many", best_of(exact_match, "cat", ["dog", "cat"])[0])


def test_oolong_scorer():
    print("\n[oolong scorer]")
    from oolong_synth_benchmark import oolong_scorer
    check("numeric gold '[7]' vs '7'", oolong_scorer("7", "[7]")[0])
    check("numeric gold in phrase", oolong_scorer("The count is 7.", "[7]")[0])
    check("label gold '['incorrect']'", oolong_scorer("Label: incorrect", "['incorrect']")[0])
    check("label rejects wrong", not oolong_scorer("correct", "['incorrect']")[0])
    check("phrase label 'more common than'",
          oolong_scorer("A is more common than B", "['more common than']")[0])


def test_niah_build():
    print("\n[niah build (synthetic, no network)]")
    from niah_benchmark import build_examples
    args = SimpleNamespace(num_samples=3, start=0, seed=1, haystack_words=500)
    exs = build_examples(args)
    check("built 3 examples", len(exs) == 3)
    ex = exs[0]
    check("needle number present in query", str(ex.answer) in ex.query)
    check("output_schema is int", ex.output_schema is int)
    check("scorer matches gold", numeric_match(ex.answer, ex.answer)[0])
    check("scorer rejects wrong guess", not numeric_match(ex.answer + 1, ex.answer)[0])


def test_streaming_build(name, module_name, kwargs):
    print(f"\n[{name} build (streaming pull, NO llm call)]")
    try:
        mod = __import__(module_name)
        args = SimpleNamespace(num_samples=2, start=0, seed=0, **kwargs)
        exs = mod.build_examples(args)
        check("pulled >=1 example", len(exs) >= 1)
        if exs:
            ex = exs[0]
            check("query is non-empty str", isinstance(ex.query, str) and len(ex.query) > 50)
            check("answer present", ex.answer not in (None, "", []))
            check("meta populated", isinstance(ex.meta, dict) and bool(ex.meta))
    except Exception as e:
        print(f"  {FAIL} {name} build raised: {type(e).__name__}: {e}")
        _failures.append(f"{name} build")


if __name__ == "__main__":
    test_scorers()
    test_oolong_scorer()
    test_niah_build()
    # Streaming pulls (small) — validate field names against the live datasets.
    test_streaming_build("oolong-synth", "oolong_synth_benchmark",
                         {"task": None, "max_context_len": 4000})
    test_streaming_build("longbench", "longbench_benchmark",
                         {"config": "narrativeqa", "f1_threshold": 0.3})

    print("\n" + ("=" * 40))
    if _failures:
        print(f"SMOKETEST FAILED: {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKETEST PASSED — no LLM was invoked.")
