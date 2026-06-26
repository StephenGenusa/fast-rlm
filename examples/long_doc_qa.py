"""
Live OOLONG-style demo (requires ANTHROPIC_API_KEY).

Asks a *semantic* question over a large structured context — the kind of task
where plain grep fails and the RLM must chunk + map sub-calls in parallel, then
reduce. Demonstrates `batch_llm_query`.

    python examples/long_doc_qa.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlm import RLM, AnthropicClient  # noqa: E402

TOPICS = {
    "sports": ["Who won the league final?", "What was the score at halftime?"],
    "cooking": ["How long do I roast a chicken?", "What temperature for bread?"],
    "finance": ["What was the closing price?", "How do dividends get taxed?"],
}


def make_log(n: int) -> tuple[str, dict[str, int]]:
    counts = {k: 0 for k in TOPICS}
    rows = []
    for i in range(n):
        topic = random.choice(list(TOPICS))
        counts[topic] += 1
        q = random.choice(TOPICS[topic])
        rows.append(f"Row {i} || User: {random.randint(1000, 9999)} || Q: {q}")
    return "\n".join(rows), counts


def main():
    context, counts = make_log(2000)
    print("ground truth counts:", counts)

    rlm = RLM(
        client=AnthropicClient("claude-sonnet-4-6"),
        sub_client=AnthropicClient("claude-haiku-4-5-20251001"),
        max_iterations=12, max_depth=1, sub_max_parallel=8, verbose=True,
    )
    result = rlm.complete(
        query=("Classify every row as one of sports/cooking/finance by its "
               "question, and report how many rows fall in each category."),
        context=context,
    )
    print("\n--- RESULT ---")
    print(result.answer)
    print("usage:", result.usage)


if __name__ == "__main__":
    main()
