"""
Provider-agnostic LLM clients for the RLM engine.

Every client returns a `Completion` carrying both the text and token usage so the
RLM can track cost across the root call and every recursive sub-call. Real
providers wrap each call in bounded retry with jittered exponential backoff and a
per-call timeout; transient 429/5xx/network blips are retried rather than
surfacing as `[sub-call error: ...]` in the REPL.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.cost_usd += other.cost_usd

    def __str__(self) -> str:
        return (
            f"calls={self.calls} in={self.input_tokens} "
            f"out={self.output_tokens} total={self.input_tokens + self.output_tokens} "
            f"cost=${self.cost_usd:.4f}"
        )


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


Message = dict  # {"role": str, "content": str}


# Editable fallback price table: model-name substring -> (USD per 1K input,
# USD per 1K output). Left empty on purpose — prices change and shipping stale
# numbers would make cost reports quietly wrong. Populate it for your models, or
# install LiteLLM (its maintained price map is used automatically when present).
# Example:  _PRICES["gpt-5-mini"] = (0.00025, 0.0020)
_PRICES: dict[str, tuple[float, float]] = {}
_warned_models: set[str] = set()


def _compute_cost(model: str, in_tok: int, out_tok: int) -> float:
    """Best-effort USD cost. Prefers LiteLLM's maintained map, then _PRICES."""
    try:
        import litellm
        pc, cc = litellm.cost_per_token(
            model=model, prompt_tokens=in_tok, completion_tokens=out_tok)
        total = float(pc + cc)
        if total > 0:
            return total
    except Exception:  # noqa - litellm absent or model unknown to it
        pass
    for key, (pin, pout) in _PRICES.items():
        if key in (model or ""):
            return in_tok / 1000 * pin + out_tok / 1000 * pout
    if model and model not in _warned_models:
        _warned_models.add(model)
        import sys
        sys.stderr.write(
            f"[rlm] no price known for model {model!r}; cost_usd=0. Install litellm or "
            "add an entry to rlm.clients._PRICES to enable dollar accounting.\n")
    return 0.0


def _retry(fn, *, retries: int, base_delay: float = 0.5, max_delay: float = 8.0):
    """Call fn(); retry on any exception with jittered exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:  # noqa
            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            time.sleep(delay * (0.5 + random.random()))  # full-ish jitter


class LLMClient(Protocol):
    model: str
    def complete(self, messages: Sequence[Message], **kwargs) -> Completion: ...


def _split_system(messages: Sequence[Message]) -> tuple[str, list[Message]]:
    system_parts, rest = [], []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            rest.append({"role": m["role"], "content": m["content"]})
    return "\n\n".join(system_parts), rest


class AnthropicClient:
    """Claude via the Anthropic Messages API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None,
                 max_tokens: int = 8192, timeout: float = 120.0, retries: int = 2):
        from anthropic import Anthropic  # lazy import
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, messages: Sequence[Message], **kwargs) -> Completion:
        system, convo = _split_system(messages)
        if not convo:
            convo = [{"role": "user", "content": "Continue."}]
        _t0 = time.time()
        resp = _retry(lambda: self._client.messages.create(
            model=self.model,
            system=system or None,
            messages=convo,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            timeout=kwargs.get("timeout", self.timeout),
        ), retries=self.retries)
        lat = time.time() - _t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        it, ot = resp.usage.input_tokens, resp.usage.output_tokens
        usage = Usage(it, ot, calls=1, cost_usd=_compute_cost(self.model, it, ot))
        return Completion(text, usage, latency_s=lat)


class OpenAIClient:
    """GPT models via the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-5-mini", api_key: str | None = None,
                 max_tokens: int | None = None, timeout: float = 120.0, retries: int = 2):
        from openai import OpenAI  # lazy import
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete(self, messages: Sequence[Message], **kwargs) -> Completion:
        _t0 = time.time()
        resp = _retry(lambda: self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_completion_tokens=kwargs.get("max_tokens", self.max_tokens),
            timeout=kwargs.get("timeout", self.timeout),
        ), retries=self.retries)
        lat = time.time() - _t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        if u:
            usage = Usage(u.prompt_tokens, u.completion_tokens, calls=1,
                          cost_usd=_compute_cost(self.model, u.prompt_tokens, u.completion_tokens))
        else:
            usage = Usage(calls=1)
        return Completion(text, usage, latency_s=lat)


class LiteLLMClient:
    """One client for many providers via LiteLLM (OpenRouter, local servers, etc.)."""

    def __init__(self, model: str, api_base: str | None = None,
                 api_key: str | None = None, max_tokens: int | None = None,
                 temperature: float | None = None, extra: dict | None = None,
                 timeout: float = 120.0, retries: int = 2):
        import litellm  # lazy import
        self._litellm = litellm
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = extra or {}
        self.timeout = timeout
        self.retries = retries

    def complete(self, messages: Sequence[Message], **kwargs) -> Completion:
        params: dict = dict(model=self.model, messages=list(messages), **self.extra)
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        mt = kwargs.get("max_tokens", self.max_tokens)
        if mt is not None:
            params["max_tokens"] = mt
        if self.temperature is not None:
            params["temperature"] = self.temperature
        params["timeout"] = kwargs.get("timeout", self.timeout)
        _t0 = time.time()
        resp = _retry(lambda: self._litellm.completion(**params), retries=self.retries)
        lat = time.time() - _t0
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        if u:
            usage = Usage(u.prompt_tokens, u.completion_tokens, calls=1,
                          cost_usd=_compute_cost(self.model, u.prompt_tokens, u.completion_tokens))
        else:
            usage = Usage(calls=1)
        return Completion(text, usage, latency_s=lat)


class MockClient:
    """Deterministic client for tests and offline demos."""

    def __init__(self, responses: Sequence[str] | None = None,
                 responder: Callable[[Sequence[Message]], str] | None = None,
                 model: str = "mock"):
        self.model = model
        self._responses = list(responses or [])
        self._responder = responder
        self._i = 0

    def complete(self, messages: Sequence[Message], **kwargs) -> Completion:
        if self._responder is not None:
            text = self._responder(messages)
        elif self._i < len(self._responses):
            text = self._responses[self._i]
            self._i += 1
        else:
            text = "FINAL(no scripted response left)"
        in_tok = sum(len(m["content"]) for m in messages) // 4
        out_tok = len(text) // 4
        return Completion(text, Usage(in_tok, out_tok, calls=1))
