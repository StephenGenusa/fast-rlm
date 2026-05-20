import fast_rlm
from pydantic import BaseModel

# Estimated time: 1 minute


class NameRCount(BaseModel):
    name: str
    r_count: int


prompt = """
You need to generate three lists — 25 fruits, 25 animals, and 25 US state names — and then count the number of times the letter 'r' (case-insensitive) appears in each name.

You MUST use 6 parallel subagent calls via asyncio.gather to generate the lists concurrently:
- Subagent 1: Generate exactly 25 fruit names.
- Subagent 2: Generate exactly 25 animal names.
- Subagent 3: Generate exactly 25 US state names.
- Subagent 4: Generate exactly 25 Indian state names.
- Subagent 5: Generate exactly 25 European country names.
- Subagent 6: Generate exactly 25 human first names.

Require each subagent's FINAL value to be a JSON list of strings by passing a JSON Schema as the second argument to llm_query:

    schema = {"type": "array", "items": {"type": "string"}}
    fruits = await llm_query("Generate exactly 25 fruit names.", schema)

After all 6 subagents return, combine the results and build a single list of
{"name": <string>, "r_count": <int>} entries — one per generated name. Pass this
list to FINAL. The runner will validate it against the required output schema.
"""

config = fast_rlm.RLMConfig()
config.primary_agent = "minimax/minimax-m2.5"
config.sub_agent = "minimax/minimax-m2.5"

data = fast_rlm.run(
    prompt,
    config=config,
    prefix="parallel_r_count",
    output_schema=list[NameRCount],
)

print("Result:", data.get("results"))
print("Usage:", data.get("usage"))
print("Log:", data.get("log_file"))
