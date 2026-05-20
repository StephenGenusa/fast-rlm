"""Demonstrates structured input AND structured output.

- Input: a dict context with named fields. The agent's initial probe will
  print a flat schema (keys + truncated previews) instead of dumping the
  whole context as a string.
- Output: a Pydantic model. The agent's FINAL value is validated against
  the corresponding JSON Schema; on validation failure the agent retries
  with the schema and error path shown to it.
"""

import fast_rlm
from pydantic import BaseModel


class Reviewer(BaseModel):
    name: str
    score: int
    summary: str


class Verdict(BaseModel):
    movie: str
    average_score: float
    consensus: str
    reviewers: list[Reviewer]


structured_context = {
    "task": (
        "Aggregate the reviews into a single verdict. Compute the average "
        "score, write a one-sentence consensus, and return per-reviewer "
        "summaries (one sentence each)."
    ),
    "movie": "The Trail of Pixels",
    "reviews": [
        {
            "name": "Asha",
            "score": 8,
            "text": "Tight pacing, great score, the third act loses some steam but lands the emotional beats.",
        },
        {
            "name": "Bo",
            "score": 6,
            "text": "Beautiful to look at but the plot is paper thin. Worth a watch on a slow night.",
        },
        {
            "name": "Cyrus",
            "score": 9,
            "text": "An instant favorite. The lead performance is magnetic and the dialogue crackles.",
        },
    ],
}

config = fast_rlm.RLMConfig()
config.primary_agent = "minimax/minimax-m2.5"
config.sub_agent = "minimax/minimax-m2.5"
config.max_depth = 1
config.max_calls_per_subagent = 6
config.max_money_spent = 0.10

result = fast_rlm.run(
    structured_context,
    config=config,
    prefix="structured_io",
    output_schema=Verdict,
)

print("\n=== RESULT ===")
print(result["results"])
print("\nLOG:", result.get("log_file"))
print("USAGE:", result.get("usage"))

# The result is JSON-compatible — Pydantic can re-parse it for type safety.
verdict = Verdict.model_validate(result["results"])
print("\nParsed verdict:", verdict)
