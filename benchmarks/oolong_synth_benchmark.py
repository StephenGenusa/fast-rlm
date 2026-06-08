"""OOLONG-synth: dense per-line reasoning + counting/aggregation.

This is the benchmark where the RLM scaffold should shine — base models score
<50% because the answer depends on (nearly) every line of the context. Answers
in the dataset are stringified lists, e.g. "[7]" or "['incorrect']".

Loaded in STREAMING mode so we never materialize the full (multi-GB) dataset —
we only pull `--num-samples` examples.

Usage:
    uv run benchmarks/oolong_synth_benchmark.py            # 3 samples
    uv run benchmarks/oolong_synth_benchmark.py -n 10
    uv run benchmarks/oolong_synth_benchmark.py -n 5 --task counting
    uv run benchmarks/oolong_synth_benchmark.py -n 5 --max-context-len 8000
"""

import ast
import itertools

from datasets import load_dataset

from _harness import (
    Example,
    base_argparser,
    config_from_args,
    llm_kwargs_from_args,
    numeric_match,
    run_benchmark,
    select_only,
    _normalize,
)


def oolong_scorer(pred, answer):
    """Score against OOLONG's stringified-list gold answer."""
    try:
        expected = ast.literal_eval(answer)
    except (ValueError, SyntaxError):
        expected = answer
    if isinstance(expected, list):
        expected = expected[0] if len(expected) == 1 else expected

    # Numeric (counting) answers: compare as numbers, tolerate phrasing.
    if isinstance(expected, bool):
        expected = str(expected)
    if isinstance(expected, (int, float)):
        return numeric_match(pred, expected)

    # Label answers (e.g. 'correct', 'more common than'): normalized match,
    # accepting the label appearing inside a longer phrase like "Label: correct".
    ne, np_ = _normalize(expected), _normalize(pred)
    ok = bool(ne) and (ne == np_ or ne in np_)
    return ok, f"pred={str(pred)[:60]!r} expected={expected!r}"


def build_examples(args) -> "list[Example]":
    ds = load_dataset("oolongbench/oolong-synth", split="test", streaming=True)

    def keep(e):
        if args.task and e.get("task_group") != args.task:
            return False
        if args.max_context_len and (e.get("context_len") or 0) > args.max_context_len:
            return False
        return True

    picked = itertools.islice(
        (e for e in ds if keep(e)),
        args.start,
        args.start + args.num_samples,
    )

    examples = []
    for e in picked:
        query = f"{e['context_window_text_with_labels']}\n\n{e['question']}"
        examples.append(
            Example(
                query=query,
                answer=e["answer"],
                meta={
                    "task": e.get("task_group"),
                    "ctx_len": e.get("context_len"),
                },
            )
        )
    return examples


def main():
    p = base_argparser(__doc__)
    p.add_argument("--task", choices=["counting", "user", "timeline"],
                   help="Only evaluate one task group.")
    p.add_argument("--max-context-len", type=int, default=None,
                   help="Skip examples whose context_len exceeds this (cheaper).")
    args = p.parse_args()

    examples = select_only(build_examples(args), args)
    run_benchmark("oolong-synth", examples, lambda pred, ex: oolong_scorer(pred, ex.answer),
                  config=config_from_args(args), prefix="oolong_synth",
                  verbose=args.verbose, concurrency=args.concurrency,
                  llm_kwargs=llm_kwargs_from_args(args))


if __name__ == "__main__":
    main()
