"""
Offline unit tests — no API keys, no JS runtime required.

These cover the pure logic of the engine (fence/FINAL parsing, schema validation
+ repair, sub-call routing/ordering, provider retry, cost accounting, and pool
mechanics via a stub). All code-execution behavior runs in the real Pyodide
sandbox and is covered by tests/test_sandbox.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm import RLM, MockClient                                  # noqa: E402
from rlm.core import extract_repl_blocks, extract_text_final     # noqa: E402


def test_fence_prefers_repl_but_falls_back_to_python():
    both = "```repl\nA\n```\n```python\nB\n```"
    assert extract_repl_blocks(both) == ["A"]                    # repl preferred
    assert extract_repl_blocks("```python\nP\n```") == ["P"]     # python fallback
    assert extract_repl_blocks("```py\nQ\n```") == ["Q"]         # py fallback
    assert extract_repl_blocks("no code here") == []


def test_balanced_final_parsing_handles_nested_parens():
    found, expr = extract_text_final('FINAL("f(x) = (a+b)*c")')
    assert found and expr == '"f(x) = (a+b)*c"'


def test_final_var_not_confused_with_final():
    found, expr = extract_text_final("FINAL_VAR(answer)")
    assert found and expr == "answer"


def test_batch_sub_handler_runs_in_parallel_and_preserves_order():
    rlm = RLM(client=MockClient(),
              sub_client=MockClient(responder=lambda m: m[-1]["content"][-1]))
    handler = rlm._make_sub_handler(depth=0)
    out = handler("batch", [{"task": "echo", "data": d} for d in ["A", "B", "C"]])
    assert out == ["A", "B", "C"]


def test_schema_validation_repairs_invalid_then_returns_object():
    schema = {"type": "object", "properties": {"value": {"type": "integer"}},
              "required": ["value"]}
    calls = {"n": 0}

    def responder(_m):
        calls["n"] += 1
        return '{"value": "nope"}' if calls["n"] == 1 else '{"value": 42}'

    rlm = RLM(client=MockClient(), sub_client=MockClient(responder=responder))
    out = rlm._make_sub_handler(0)("query", {"task": "x", "data": None, "schema": schema})
    try:
        import jsonschema  # noqa
        assert out == {"value": 42} and calls["n"] == 2
    except ImportError:
        assert isinstance(out, dict)


def test_tool_dispatch_offline():
    """The tool sub-kind routes to the registered host callable (no sandbox)."""
    def add(a, b):
        "add two numbers"
        return a + b
    rlm = RLM(client=MockClient(), tools={"add": add})
    handler = rlm._make_sub_handler(0)
    assert handler("tool", {"name": "add", "args": [2, 3], "kwargs": {}}) == 5
    assert "unknown tool" in str(handler("tool", {"name": "nope", "args": [], "kwargs": {}}))


def test_retry_recovers_after_transient_failures():
    from rlm.clients import _retry
    n = {"i": 0}

    def flaky():
        n["i"] += 1
        if n["i"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert _retry(flaky, retries=3, base_delay=0.001) == "ok" and n["i"] == 3


def test_cost_accounting_table_and_aggregation():
    from rlm.clients import _compute_cost, _PRICES, Usage
    _PRICES["zzz-test-model"] = (1.0, 2.0)  # $1/1K in, $2/1K out
    try:
        assert abs(_compute_cost("zzz-test-model", 1000, 500) - 2.0) < 1e-9
        assert _compute_cost("totally-unknown-model-xyz", 1000, 1000) == 0.0
        u = Usage(cost_usd=1.5); u.add(Usage(cost_usd=0.25))
        assert abs(u.cost_usd - 1.75) < 1e-9
    finally:
        _PRICES.pop("zzz-test-model", None)


def test_litellm_cost_estimation():
    """When LiteLLM is installed, cost comes from its maintained price map."""
    try:
        import litellm  # noqa
    except ImportError:
        print("SKIP (litellm not installed)"); return
    from rlm.clients import _compute_cost
    cost = _compute_cost("gpt-4o-mini", 1000, 1000)
    assert cost > 0, cost


class _StubSandbox:
    """Minimal sandbox-like object for testing pool mechanics without a runtime."""
    def __init__(self): self.reset_calls = 0
    def start(self): pass
    def reset(self): self.reset_calls += 1
    def close(self): pass


def test_sandbox_pool_reuse_and_concurrency():
    from rlm import SandboxPool
    pool = SandboxPool(factory=_StubSandbox, size=2)
    a = pool.acquire(); b = pool.acquire()      # held -> distinct under the cap
    assert a is not b
    pool.release(a)
    assert a.reset_calls == 1                    # cleared on release
    assert pool.acquire() is a                   # warm reuse of the released one
    pool.close()


def test_tracer_span_recording_and_otel():
    from rlm import Tracer
    seen = []
    t = Tracer(on_span=seen.append)
    with t.span("root_turn", "turn 0", model="m") as sp:
        sp.input_tokens, sp.output_tokens = 5, 3
    assert len(t.spans) == 1 and seen and seen[0].kind == "root_turn"
    assert t.spans[0].latency_s >= 0
    d = t.spans[0].to_otel()
    assert d["attributes"]["gen_ai.request.model"] == "m" and d["trace_id"] == t.trace_id


def test_handler_emits_sub_and_tool_spans():
    from rlm import Tracer
    def add(a, b):
        "add"
        return a + b
    t = Tracer()
    rlm = RLM(client=MockClient(), sub_client=MockClient(responder=lambda m: "R"),
              tools={"add": add})
    h = rlm._make_sub_handler(0, tracer=t, parent_box=["P"])
    assert h("query", {"task": "q", "data": None, "schema": None}) == "R"
    assert h("tool", {"name": "add", "args": [2, 3], "kwargs": {}}) == 5
    kinds = [sp.kind for sp in t.spans]
    assert "sub_query" in kinds and "tool" in kinds
    assert all(sp.parent_id == "P" for sp in t.spans)


def test_trace_content_can_be_disabled():
    from rlm import Tracer
    t = Tracer(trace_content=False)
    with t.span("x", "x") as sp:
        sp.content = t.content(prompt="secret")
    assert t.spans[0].content == {}


def test_on_span_callback_can_reenter_tracer_without_deadlock():
    from rlm import Tracer
    t = Tracer()
    t._on_span = lambda sp: t.to_list()   # callback reads the tracer (would deadlock if held)
    with t.span("x", "x"):
        pass
    assert len(t.to_list()) == 1


def test_trace_content_opt_in_and_redaction():
    from rlm import Tracer
    assert Tracer().content(prompt="secret") == {}                 # off by default
    assert Tracer(trace_content=True).content(prompt="hi")["prompt"] == "hi"
    t = Tracer(trace_content=True, redact=lambda s: s.replace("KEY123", "***"))
    assert t.content(prompt="api KEY123")["prompt"] == "api ***"


def test_usage_aggregation_thread_safe_under_batch():
    rlm = RLM(client=MockClient(), sub_client=MockClient(responder=lambda m: "x"))
    handler = rlm._make_sub_handler(0)
    n = 50
    handler("batch", [{"task": "t", "data": i, "schema": None} for i in range(n)])
    assert rlm.usage.calls == n   # no lost updates under the parallel fan-out


def test_llm_query_is_always_leaf_even_at_depth2():
    from rlm import Tracer
    t = Tracer()
    rlm = RLM(client=MockClient(), sub_client=MockClient(responder=lambda m: "L"), max_depth=2)
    h = rlm._make_sub_handler(0, tracer=t, parent_box=[None])
    assert h("query", {"task": "q", "data": None, "schema": None}) == "L"
    assert [sp.kind for sp in t.spans] == ["sub_query"]   # leaf, never a child rlm


def test_rlm_query_falls_back_to_leaf_at_depth_budget():
    from rlm import Tracer
    t = Tracer()
    rlm = RLM(client=MockClient(), sub_client=MockClient(responder=lambda m: "R"), max_depth=1)
    h = rlm._make_sub_handler(0, tracer=t, parent_box=[None])
    assert h("rlm_query", {"task": "q", "data": None, "schema": None}) == "R"
    assert [sp.kind for sp in t.spans] == ["sub_query"]   # at the budget -> leaf fallback


def test_rlm_batch_falls_back_to_leaves_at_depth1():
    rlm = RLM(client=MockClient(), sub_client=MockClient(responder=lambda m: "x"), max_depth=1)
    h = rlm._make_sub_handler(0)
    out = h("rlm_batch", [{"task": "t", "data": i, "schema": None} for i in range(3)])
    assert out == ["x", "x", "x"]


def test_prompt_advertises_rlm_query_only_when_recursion_enabled():
    from rlm.prompts import build_messages
    off = build_messages("q", "meta", recursive_enabled=False)[0]["content"]
    on = build_messages("q", "meta", recursive_enabled=True)[0]["content"]
    assert "rlm_query" not in off
    assert "rlm_query" in on and "batch_rlm_query" in on


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
