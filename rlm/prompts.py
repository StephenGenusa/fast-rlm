"""
Prompts for the RLM root model.

Design note — the failure mode this fixes
------------------------------------------
A naive "recursive prompt" pastes the query and the long context into the same
string and feeds it to the model, so the model cannot tell its *instructions*
apart from the *data* it is meant to operate on, and it gets confused. RLM
removes that ambiguity structurally: the root model only ever sees the QUERY in
its prompt. The CONTEXT lives in a REPL as a variable named `context` that the
model inspects with code. There is never a turn where query and context are
concatenated into one blob. The prompt below leans on that invariant hard.
"""

ROOT_SYSTEM_PROMPT = """\
You answer a QUERY about a CONTEXT you cannot see directly. The context may be \
far larger than any context window. It lives in a Python REPL as a variable \
named `context` (a str or a dict). You interact with it ONLY by writing code.

Hard rules
- Your prompt contains the QUERY and metadata about `context` (its type and \
size). It never contains the context itself. Do not pretend to have read \
content you have not printed.
- To read context, print slices of it: `print(context[:2000])`. Output is \
TRUNCATED before it reaches you, so never try to print the whole thing.
- Never paste large strings into a sub-call as part of an instruction. Pass the \
data as a separate argument so the sub-model can tell task from data — exactly \
the discipline you are operating under.

The REPL gives you these functions:
- `{aw}llm_query(task, data=None, schema=None)` -> a sub-model answer. `task` is \
the instruction; `data` is the (sliced) context to operate on. Keep task and \
data separate. If `schema` (a JSON-schema dict) is given, the result is \
validated and returned as a Python object.
- `{aw}batch_llm_query(jobs)` -> run many `llm_query` calls IN PARALLEL and get \
results in order. `jobs` is a list of dicts: \
`[{{"task": "...", "data": chunk}}, ...]`. Always prefer this over a Python loop \
of `llm_query`; serial sub-calls are the main source of slowness.
- `FINAL(value)` -> end the run and return `value` (str, dict, list, number...).
- The REPL is persistent like a notebook: variables survive across your turns. \
Never overwrite or delete `context`.

Recommended strategy
1. Peek: print `context[:2000]` (and a slice from the middle/end) to learn its \
shape and any structure (headers, JSON, rows).
2. Narrow: use plain Python (`in`, regex, `.split`) to cut the search space \
before spending a sub-call. Programmatic answers (counting, diffs, exact \
extraction) should be done in code, not delegated.
3. Map: when semantics are needed, chunk the context and `batch_llm_query` one \
focused task per chunk; store answers in a list as a buffer.
4. Reduce: combine the buffer (with a final `llm_query` if needed) and `FINAL` it.

Write Python in fenced blocks tagged `repl` (a ```python/```py block is also accepted as a fallback if you forget the tag):
```repl
print(type(context), len(context) if isinstance(context, str) else list(context)[:10])
```
You may also call `FINAL(...)` directly inside a repl block. Think briefly, then \
ACT in code every turn — do not narrate a plan without running it.{async_note}{tools_note}{recursive_note}
"""

DEFAULT_QUERY = (
    "Read the context and answer any question or follow any instruction it contains."
)


def _render_system(async_subcalls: bool, tools_doc: str = "",
                    recursive_enabled: bool = False) -> str:
    aw = "await " if async_subcalls else ""
    note = ("\n\nNOTE: sub-calls are asynchronous — you MUST write "
            "`await llm_query(...)` and `await batch_llm_query(...)`."
            if async_subcalls else
            "\n\nNOTE: sub-calls are synchronous — call `llm_query(...)` directly "
            "(do NOT write await).")
    tools_note = ""
    if tools_doc:
        tools_note = (f"\n\nHost tools are available (call as `{aw}tool(name, ...)`, "
                      "same await rule as sub-calls). Results are JSON values:\n"
                      f"{tools_doc}")
    recursive_note = ""
    if recursive_enabled:
        recursive_note = (
            f"\n\nFor sub-tasks that themselves need to explore a large chunk, you can "
            f"call `{aw}rlm_query(task, data=..., schema=None)` (and "
            f"`{aw}batch_rlm_query(jobs)`): these spawn a full recursive child RLM with "
            "its own REPL over `data`, rather than a single model call. They cost more "
            "than `llm_query` — prefer `llm_query` for simple extraction/labeling and "
            "reserve `rlm_query` for sub-tasks that need their own decomposition.")
    return ROOT_SYSTEM_PROMPT.format(aw=aw, async_note=note, tools_note=tools_note,
                                     recursive_note=recursive_note)


def build_messages(query: str, context_meta: str, async_subcalls: bool = True,
                   tools_doc: str = "", recursive_enabled: bool = False) -> list[dict]:
    """Initial conversation: system rules + the query + context metadata only."""
    return [
        {"role": "system", "content": _render_system(async_subcalls, tools_doc, recursive_enabled)},
        {
            "role": "user",
            "content": (
                f"QUERY:\n{query}\n\n"
                f"CONTEXT (in the REPL as `context`, not shown here):\n{context_meta}\n\n"
                "Begin by inspecting `context`. Write a repl block now."
            ),
        },
    ]


def step_prompt(query: str, iteration: int, force_final: bool = False) -> dict:
    if force_final:
        return {
            "role": "user",
            "content": (
                "You are out of iterations. Using only what you have already "
                f"gathered, call FINAL(...) now to answer: {query!r}"
            ),
        }
    if iteration == 0:
        return {
            "role": "user",
            "content": "You have not inspected `context` yet. Inspect it before answering.",
        }
    return {
        "role": "user",
        "content": (
            "Continue. Use the REPL (and parallel sub-calls where useful) to make "
            f"progress on: {query!r}. Call FINAL(...) when confident."
        ),
    }
