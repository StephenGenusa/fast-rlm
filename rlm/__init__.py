"""
rlm — a working Recursive Language Model engine with a mandatory code sandbox.

Quick start (sandboxed; requires Deno or Node + `npm install pyodide`):
    from rlm import RLM, LiteLLMClient
    rlm = RLM(
        client=LiteLLMClient("openrouter/anthropic/claude-3.5-sonnet"),
        sub_client=LiteLLMClient("openrouter/anthropic/claude-3.5-haiku"),
    )
    out = rlm.complete(query="What is the magic number?", context=huge_string)
    print(out.answer, out.usage)
"""

from .clients import (
    AnthropicClient, Completion, LiteLLMClient, LLMClient, MockClient,
    OpenAIClient, Usage,
)
from .core import RLM, RLMResult
from .tracing import Span, Tracer, load_jsonl
from .tui import run_tui, live_tui
from .sandbox import PyodideSandbox, Sandbox, SandboxPool, SandboxResult

__all__ = [
    "RLM", "RLMResult", "Span", "Tracer", "load_jsonl", "run_tui", "live_tui",
    "PyodideSandbox", "Sandbox", "SandboxPool", "SandboxResult",
    "AnthropicClient", "OpenAIClient", "LiteLLMClient", "MockClient",
    "LLMClient", "Completion", "Usage",
]
__version__ = "0.2.0"
