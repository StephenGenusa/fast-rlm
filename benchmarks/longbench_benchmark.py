"""LongBench: realistic long-context QA (NarrativeQA, HotpotQA, ...).

Cheap, well-known QA with multiple acceptable answers; scored with token-level
F1 (SQuAD/LongBench style), 'correct' when F1 >= --f1-threshold.

Loaded in STREAMING mode so we only pull `--num-samples` examples instead of the
whole config.

Usage:
    uv run benchmarks/longbench_benchmark.py                  # 3 narrativeqa samples
    uv run benchmarks/longbench_benchmark.py -n 10 --config hotpotqa
"""

import itertools

from datasets import load_dataset

from _harness import (
    Example,
    base_argparser,
    best_of,
    config_from_args,
    judge_correct,
    llm_kwargs_from_args,
    f1_score,
    run_benchmark,
    select_only,
)


def build_examples(args) -> "list[Example]":
    ds = load_dataset(
        "THUDM/LongBench", args.config, split="test",
        streaming=True, trust_remote_code=True,
    )
    picked = itertools.islice(ds, args.start, args.start + args.num_samples)

    examples = []
    for e in picked:
        query = f"{e['input']}\n\n{e['context']}"
        examples.append(
            Example(
                query=query,
                answer=e["answers"],  # list of acceptable answers
                meta={"dataset": e.get("dataset", args.config),
                      "length": e.get("length"),
                      "question": e["input"]},
            )
        )
    return examples


def main():
    p = base_argparser(__doc__)
    p.add_argument("--config", default="narrativeqa",
                   help="LongBench sub-config (default: narrativeqa).")
    p.add_argument("--f1-threshold", type=float, default=0.3,
                   help="F1 above which an answer counts as correct (default: 0.3).")
    args = p.parse_args()

    if args.judge:
        def scorer(pred, ex):
            return judge_correct(pred, ex.answer, model=args.judge_model,
                                 question=ex.meta.get("question"))
    else:
        def scorer(pred, ex):
            return best_of(lambda p, a: f1_score(p, a, args.f1_threshold), pred, ex.answer)

    examples = select_only(build_examples(args), args)
    run_benchmark(f"longbench/{args.config}", examples, scorer,
                  config=config_from_args(args), prefix=f"longbench_{args.config}",
                  verbose=args.verbose, concurrency=args.concurrency,
                  llm_kwargs=llm_kwargs_from_args(args))


if __name__ == "__main__":
    main()
