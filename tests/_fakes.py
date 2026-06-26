"""Shared test doubles: a scripted sandbox, a fake RLM, and fake provider SDKs.

These are the only mocks in the suite — used at true external seams (the WASM
subprocess for offline branch tests; provider SDKs to avoid the network).
"""
from __future__ import annotations

import sys
import types


# --- a sandbox stub implementing the Sandbox protocol (offline) -------------
class ScriptedSandbox:
    async_subcalls = False

    def __init__(self, results=None, on_run=None):
        self._results = list(results or [])
        self._on_run = on_run
        self.started = self.closed = False
        self.context = None
        self.reset_calls = 0

    def start(self): self.started = True
    def set_context(self, c): self.context = c
    def reset(self): self.reset_calls += 1
    def clone(self): return ScriptedSandbox(list(self._results), self._on_run)
    def close(self): self.closed = True

    def run(self, code, sub_handler):
        from rlm.sandbox import SandboxResult
        if self._on_run is not None:
            return self._on_run(code, sub_handler)
        if self._results:
            r = self._results.pop(0)
            return r(sub_handler) if callable(r) else r
        return SandboxResult("(no scripted result)", None, False)


def sresult(stdout="", final=None, has_final=False):
    from rlm.sandbox import SandboxResult
    return SandboxResult(stdout, final, has_final)


# --- fake RLM for CLI / TUI / live tests ------------------------------------
class FakeRLM:
    def __init__(self, answer="OK", spans=None, trace_content=True, trace_dir="rlm_traces"):
        self.answer = answer
        self._spans = spans or []
        self.on_span = None
        self.trace_content = trace_content
        self.trace_dir = trace_dir

    def complete(self, query, context, **kw):
        from rlm.core import RLMResult
        from rlm.clients import Usage
        for sp in self._spans:
            if self.on_span:
                self.on_span(sp)
        return RLMResult(self.answer, Usage(1, 2, 1, 0.0), 1, [], "tid")


# --- fake provider SDK responses --------------------------------------------
def _obj(**kw):
    o = types.SimpleNamespace(**kw)
    return o


def fake_anthropic_module(text="A", in_tok=11, out_tok=7, record=None):
    """Inject a fake `anthropic` module so AnthropicClient can be constructed."""
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            if record is not None:
                record.update(kwargs)
            return _obj(content=[_obj(type="text", text=text)],
                        usage=_obj(input_tokens=in_tok, output_tokens=out_tok))

    class Anthropic:
        def __init__(self, *a, **k): self.messages = _Messages()

    mod.Anthropic = Anthropic
    sys.modules["anthropic"] = mod
    return mod


def fake_openai_module(text="O", in_tok=13, out_tok=4, record=None):
    mod = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            if record is not None:
                record.update(kwargs)
            return _obj(choices=[_obj(message=_obj(content=text))],
                        usage=_obj(prompt_tokens=in_tok, completion_tokens=out_tok))

    class OpenAI:
        def __init__(self, *a, **k):
            self.chat = _obj(completions=_Completions())

    mod.OpenAI = OpenAI
    sys.modules["openai"] = mod
    return mod
