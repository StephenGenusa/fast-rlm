# Tools

Inside the REPL the root agent has two built-in tools — `llm_query` and `FINAL` — and may also receive user-defined Python functions as tools. There is no separate tool-calling API: tools are just callables in the REPL namespace, invoked exactly like regular Python functions.

This page covers three things:

1. [Input / Output Signatures](#input-output-signatures) — what the model sees about your tool.
2. [Tool Calling Process](#tool-calling-process) — how a tool registered in Python ends up callable inside the REPL, and how sub-agents inherit (or don't inherit) tools.
3. [Environment Variables](#environment-variables) — how to hand credentials to a tool without exposing them to the model.

---

## Input / Output Signatures

When you register a tool, the model is shown the **function name, its parameters (with type hints and defaults), and its docstring** — and *only* that. The function body is not pasted into the prompt.

```python
def filter_short(items: list[str], max_len: int = 20) -> list[str]:
    """Return only items shorter than max_len characters."""
    return [x for x in items if len(x) < max_len]
```

In the agent's initial probe this surfaces under an **"Available tools"** section as something like:

```text
filter_short(items: list[str], max_len: int = 20) -> list[str]
    Return only items shorter than max_len characters.
```

Practical consequences:

- **Type hints and docstrings are your tool's prompt.** Treat them as the contract the model reads. Vague types (`Any`, missing return type, no docstring) make the tool harder for the agent to use correctly.
- **Side effects belong in the docstring.** If the tool writes a file, charges money, or sends a request, say so — the model can't infer it from the signature.
- **The agent can `inspect.getsource(tool_name)` if it really needs the body.** It will rarely do this unsolicited, but the option exists for debugging.
- **Outputs are returned as Python objects.** Whatever the function returns lives as a regular Python value in the REPL — no serialization round-trip. Returning a `dict`, `list`, dataframe, etc. is fine as long as it can sit in the Pyodide namespace.

A tool with a richer signature:

```python
def fetch_arxiv(arxiv_id: str, sections: list[str] | None = None) -> dict:
    """Fetch an arXiv paper by ID.

    Args:
        arxiv_id: e.g. "2512.24601".
        sections: optional list of section titles to keep. If None, returns all sections.

    Returns:
        dict with keys: "title" (str), "abstract" (str), "sections" (dict[str, str]).
    """
    ...
```

**Because everything is recursive, Main agents can also create subagents and ask for specific data structures as output**

---

## Tool Calling Process

### 1. Register tools at run time

Pass a list of Python callables to `fast_rlm.run(...)`:

```python
import fast_rlm

result = fast_rlm.run(
    "Pick the short titles from the list and summarise them.",
    tools=[filter_short],
)
```

Internally, fast-rlm extracts each tool's source with `inspect.getsource` and re-executes it inside the root agent's Pyodide REPL before initialisation. After that, `filter_short` is a regular function in the REPL namespace.

### 2. The agent calls them like normal Python

The agent just writes code:

```python
short = filter_short(context["titles"], max_len=30)
print(short[:5])
```

…which executes inside the REPL and returns the result on the next turn.

Note that these tools are not available when the LLM is generating tokens through the chat.completions.create API. These are strictly REPL tools available as python functions.


### 3. Sub-agents do NOT automatically inherit tools

A sub-agent spawned via `llm_query` starts with a **fresh REPL**. None of the parent's tools (or REPL state) are carried over. To give a child a tool, the parent must pass it explicitly via the `tools=[...]` keyword to `llm_query`:

```python
# Inside the root REPL
result = await llm_query(
    "From this chunk, keep only the titles shorter than 30 chars.",
    tools=[filter_short],
)
```

This rule applies to user-registered tools *and* to functions the parent agent defined itself in its own REPL — agents can `def my_helper(...)` mid-run and hand `my_helper` down the same way.

### 4. Tools must be self-contained

Because each Pyodide REPL is isolated, a tool cannot rely on anything from its original definition site:

- **Do imports inside the function body.** `import os` at module top-level won't be available in the child REPL.
- **Don't close over outer variables.** A tool referencing `MY_GLOBAL` from the surrounding module will raise `NameError` once it lands in a sub-agent.

```python
def search_web(query: str, top_k: int = 5) -> list[dict]:
    """Search the web via Tavily and return the top results."""
    import os, urllib.request, json  # imports live INSIDE the function
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps({"query": query, "max_results": top_k}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    return json.loads(urllib.request.urlopen(req).read())["results"]
```


---

## Environment Variables

Most non-trivial tools need credentials or configuration (API keys, base URLs, account IDs) that you **do not** want to expose to the model. fast-rlm provides an `env_variables` kwarg on `fast_rlm.run(...)` for this:

```python
import os
import fast_rlm

def search_web(query: str, top_k: int = 5) -> list[dict]:
    """Search the web via Tavily and return the top results."""
    import os, urllib.request, json
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps({"query": query, "max_results": top_k}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    return json.loads(urllib.request.urlopen(req).read())["results"]

result = fast_rlm.run(
    "Find three recent papers on recursive language models.",
    tools=[search_web],
    env_variables={"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
)
```

### Behavior

- `env_variables` must be a `dict[str, str]`.
- Each entry is injected into `os.environ` inside **every** Pyodide REPL spawned by the run — the root agent and all sub-agents. Sub-agents inherit env vars automatically (unlike tools).
- They are **not** set on the host Deno process and never appear in prompts, logs, or model context. The model only sees a tool's signature + docstring, so the key stays hidden as long as your tool doesn't `print` or `return` it.
- Tools read them with the normal `os.environ["..."]`. Remember to do the `import os` inside the tool body (see the [self-containment rule](#4-tools-must-be-self-contained)).

!!! warning "Don't echo secrets back to the model"
    If your tool prints or returns the secret, it will end up in the agent's REPL output, the JSONL log, and potentially the next prompt. Treat the env var like any credential — read it, use it, don't surface it.
