"""NIAH: needle-in-a-haystack (synthetic, no dataset download).

A cheap regression / sanity floor — frontier models hit ~90%+ here, so this
won't differentiate good agents, but it's free to generate and catches gross
breakage (e.g. the agent can't retrieve a single fact from a long context).

We hide a "magic number" for a random keyword at a random depth in a wall of
filler text and ask the agent to retrieve it. Scored by numeric match.

Usage:
    uv run benchmarks/niah_benchmark.py                 # 3 needles, ~4k-word haystack
    uv run benchmarks/niah_benchmark.py -n 10 --haystack-words 20000
    uv run benchmarks/niah_benchmark.py --model acp:opencode   # drive an ACP agent
"""

import random

from _harness import (
    Example,
    base_argparser,
    config_from_args,
    llm_kwargs_from_args,
    numeric_match,
    run_benchmark,
    select_only,
)

_FILLER = (
    "The quarterly logistics report noted steady throughput across all regional hubs. "
    "Operations remained nominal and no anomalies were flagged during the review period. "
    "Staff rotations proceeded on schedule and inventory levels stayed within tolerance. "
    "Routine maintenance was completed without incident and uptime targets were met. "
)
_KEYWORDS = ["aurora", "basalt", "cobalt", "dynamo", "ember", "falcon",
             "granite", "harbor", "ivory", "jasper", "kelvin", "lumen"]


def _make_haystack(words: int, rng: random.Random) -> str:
    sentences = []
    filler_sentences = _FILLER.split(". ")
    count = 0
    while count < words:
        s = rng.choice(filler_sentences).strip()
        sentences.append(s + ".")
        count += len(s.split())
    return " ".join(sentences)


def build_examples(args) -> "list[Example]":
    rng = random.Random(args.seed)
    examples = []
    for _ in range(args.num_samples):
        keyword = rng.choice(_KEYWORDS)
        magic = rng.randint(10000, 99999)
        needle = f"The special magic number for {keyword} is {magic}."

        haystack = _make_haystack(args.haystack_words, rng)
        tokens = haystack.split()
        pos = rng.randint(0, len(tokens))  # random depth
        tokens[pos:pos] = needle.split()
        context = " ".join(tokens)

        query = (
            f"{context}\n\n"
            f"What is the special magic number for {keyword}? "
            f"Answer with the number only."
        )
        examples.append(
            Example(
                query=query,
                answer=magic,
                output_schema=int,
                meta={"keyword": keyword, "words": args.haystack_words},
            )
        )
    return examples


def main():
    p = base_argparser(__doc__)
    p.add_argument("--haystack-words", type=int, default=4000,
                   help="Approx. filler words around the needle (default: 4000).")
    args = p.parse_args()

    examples = select_only(build_examples(args), args)
    run_benchmark("niah", examples, lambda pred, ex: numeric_match(pred, ex.answer),
                  config=config_from_args(args), prefix="niah",
                  verbose=args.verbose, concurrency=args.concurrency,
                  llm_kwargs=llm_kwargs_from_args(args))


if __name__ == "__main__":
    main()
