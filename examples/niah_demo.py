"""
Live needle-in-a-haystack demo (requires ANTHROPIC_API_KEY).

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/niah_demo.py

Builds a context far larger than any context window, stores it as a REPL
variable, and lets the RLM find the planted number. Note that `context` is
NEVER placed in the prompt — the root model only receives the query and the
context's metadata, then explores with code.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlm import RLM, AnthropicClient  # noqa: E402


def make_haystack(num_lines: int, needle: str) -> str:
    words = ["blah", "random", "text", "data", "content", "info", "sample"]
    lines = [" ".join(random.choice(words) for _ in range(random.randint(3, 8)))
             for _ in range(num_lines)]
    pos = random.randint(num_lines // 3, 2 * num_lines // 3)
    lines[pos] = f"the magic number is {needle}"
    print(f"planted {needle!r} at line {pos:,} of {num_lines:,}")
    return "\n".join(lines)


def main():
    needle = str(random.randint(1_000_000, 9_999_999))
    context = make_haystack(1_000_000, needle)  # ~ tens of millions of chars

    rlm = RLM(
        client=AnthropicClient("claude-sonnet-4-6"),
        sub_client=AnthropicClient("claude-haiku-4-5-20251001"),
        max_iterations=10,
        max_depth=1,
        verbose=True,
    )
    result = rlm.complete(query="What is the magic number?", context=context)

    print("\n--- RESULT ---")
    print("answer   :", result.answer)
    print("expected :", needle)
    print("correct  :", str(needle) in str(result.answer))
    print("iters    :", result.iterations)
    print("usage    :", result.usage)


if __name__ == "__main__":
    main()
