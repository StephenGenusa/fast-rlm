"""Client tests — provider clients via mocked SDKs; pure helpers for real."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _fakes

from rlm.clients import (AnthropicClient, OpenAIClient, LiteLLMClient, MockClient,  # noqa: E402
                         Usage, Completion, _retry, _split_system, _compute_cost, _PRICES)
from _fakes import fake_anthropic_module, fake_openai_module  # noqa: E402


def test_split_system_separates_system_from_convo():
    sys_text, convo = _split_system([
        {"role": "system", "content": "S1"}, {"role": "system", "content": "S2"},
        {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}])
    assert sys_text == "S1\n\nS2"
    assert convo == [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]


def test_anthropic_client_shapes_request_and_extracts(monkeypatch=None):
    rec = {}
    fake_anthropic_module(text="hi", in_tok=11, out_tok=7, record=rec)
    c = AnthropicClient(model="claude-x", api_key="k", max_tokens=512)
    comp = c.complete([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert comp.text == "hi"
    assert comp.usage.input_tokens == 11 and comp.usage.output_tokens == 7 and comp.usage.calls == 1
    assert comp.latency_s >= 0
    assert rec["system"] == "S" and rec["messages"] == [{"role": "user", "content": "U"}]
    assert rec["max_tokens"] == 512 and "timeout" in rec


def test_anthropic_injects_user_turn_when_only_system():
    fake_anthropic_module(text="x")
    c = AnthropicClient(model="m", api_key="k")
    c.complete([{"role": "system", "content": "only system"}])  # must not raise


def test_openai_client_shapes_request_and_extracts():
    rec = {}
    fake_openai_module(text="ok", in_tok=13, out_tok=4, record=rec)
    c = OpenAIClient(model="gpt-x", api_key="k")
    comp = c.complete([{"role": "user", "content": "U"}])
    assert comp.text == "ok" and comp.usage.input_tokens == 13 and comp.usage.output_tokens == 4
    assert comp.latency_s >= 0 and rec["model"] == "gpt-x" and "timeout" in rec


def test_litellm_client_via_patched_completion():
    import litellm, types
    rec = {}

    def fake_completion(**kwargs):
        rec.update(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="L"))],
            usage=types.SimpleNamespace(prompt_tokens=9, completion_tokens=3))

    orig = litellm.completion
    litellm.completion = fake_completion
    try:
        c = LiteLLMClient("gpt-4o-mini", api_base="http://x", api_key="k")
        comp = c.complete([{"role": "user", "content": "U"}])
        assert comp.text == "L" and comp.usage.input_tokens == 9
        assert comp.usage.cost_usd > 0          # litellm price map applied
        assert rec["api_base"] == "http://x" and "timeout" in rec
    finally:
        litellm.completion = orig


def test_retry_exhausts_and_reraises(monkeypatch=None):
    import rlm.clients as clients
    slept = []
    orig_sleep = clients.time.sleep
    clients.time.sleep = lambda s: slept.append(s)
    try:
        calls = {"n": 0}

        def always_fail():
            calls["n"] += 1
            raise RuntimeError("boom")

        try:
            _retry(always_fail, retries=2, base_delay=0.01)
            assert False, "should re-raise"
        except RuntimeError:
            pass
        assert calls["n"] == 3 and len(slept) == 2     # 1 try + 2 retries; slept twice
    finally:
        clients.time.sleep = orig_sleep


def test_mockclient_responses_then_default_and_responder():
    m = MockClient(responses=["a", "b"])
    assert m.complete([{"role": "user", "content": "x"}]).text == "a"
    assert m.complete([{"role": "user", "content": "x"}]).text == "b"
    assert "no scripted response" in m.complete([{"role": "user", "content": "x"}]).text
    r = MockClient(responder=lambda msgs: msgs[-1]["content"].upper())
    assert r.complete([{"role": "user", "content": "hi"}]).text == "HI"


def test_compute_cost_unknown_warns_once():
    import rlm.clients as clients
    clients._warned_models.discard("brand-new-xyz")
    assert _compute_cost("brand-new-xyz", 10, 10) == 0.0
    assert "brand-new-xyz" in clients._warned_models


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa
                failures += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\n" + ("ALL TESTS PASSED" if not failures else f"{failures} FAILURE(S)"))
    sys.exit(1 if failures else 0)
