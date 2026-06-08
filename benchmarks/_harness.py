"""Shared benchmark harness for fast-rlm.

Goal (v1): given a small number of samples, did the RLM get the answer right,
and what did it cost? No budget sweeps, no non-RLM baseline yet — just
correctness + token/cost accounting so we can compare agents and validate
prompts/techniques cheaply.

Each benchmark builds a list of `Example`s and a `scorer`, then calls
`run_benchmark(...)`. Bump `--num-samples` to scale up once a config looks good.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import string
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import fast_rlm
from fast_rlm import RLMConfig


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    query: "str | dict"
    answer: Any
    # Per-example schema overrides the benchmark-wide one (e.g. counting => int).
    output_schema: Any = None
    # Free-form tags shown in the report (e.g. {"task_group": "counting"}).
    meta: dict = field(default_factory=dict)


# A scorer maps (prediction, Example) -> (is_correct, detail_str). It gets the
# whole Example so judges can use ex.answer, ex.meta (e.g. the question), etc.
Scorer = Callable[[Any, "Example"], "tuple[bool, str]"]


# --------------------------------------------------------------------------- #
# Scoring helpers (no LLM judge — keep it cheap and deterministic)
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: Any, answer: Any) -> "tuple[bool, str]":
    ok = _normalize(pred) == _normalize(answer)
    return ok, f"pred={pred!r} expected={answer!r}"


def numeric_match(pred: Any, answer: Any, tol: float = 0.0) -> "tuple[bool, str]":
    """Compare as numbers. Pulls the first number out of strings if needed."""
    def to_num(x: Any) -> Optional[float]:
        if isinstance(x, (int, float)):
            return float(x)
        m = re.search(r"-?\d+(?:\.\d+)?", str(x))
        return float(m.group()) if m else None

    p, a = to_num(pred), to_num(answer)
    if p is None or a is None:
        return False, f"pred={pred!r} expected={answer!r} (non-numeric)"
    ok = abs(p - a) <= tol
    return ok, f"pred={p} expected={a} tol={tol}"


def f1_score(pred: Any, answer: Any, threshold: float = 0.5) -> "tuple[bool, str]":
    """Token-level F1 (SQuAD/LongBench style); 'correct' if F1 >= threshold."""
    pred_toks = _normalize(pred).split()
    ans_toks = _normalize(answer).split()
    if not pred_toks or not ans_toks:
        ok = pred_toks == ans_toks
        return ok, f"f1={1.0 if ok else 0.0:.2f}"
    common = Counter(pred_toks) & Counter(ans_toks)
    same = sum(common.values())
    if same == 0:
        return False, "f1=0.00"
    precision = same / len(pred_toks)
    recall = same / len(ans_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= threshold, f"f1={f1:.2f}"


def llm_judge(
    prediction: Any,
    references: Any,
    *,
    model: str = "minimax/minimax-m3",
    question: Optional[str] = None,
    timeout: int = 120,
) -> dict:
    """Ask `model` to grade a candidate answer on two axes vs the reference(s).

    Returns {"correct": bool, "verbose": bool, "reason": str}. `correct` is judged
    leniently on meaning (the fix for token-F1 punishing wordy-but-right answers);
    `verbose` flags whether the answer is much longer than a concise reply needs.
    Uses the same OpenAI-compatible endpoint as fast-rlm. Fails closed on error.
    """
    api_key = os.environ.get("RLM_MODEL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("RLM_MODEL_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        return {"correct": False, "verbose": False, "reason": "judge error: RLM_MODEL_API_KEY not set"}

    refs = references if isinstance(references, list) else [references]
    ref_str = " | ".join(str(r) for r in refs)
    q = f"Question: {question}\n\n" if question else ""
    user = (
        f"{q}Reference answer(s): {ref_str}\n\n"
        f"Candidate answer: {prediction}\n\n"
        "Grade the candidate on two independent axes:\n"
        "1) CORRECT — does it give the same essential answer as any reference? Judge meaning, "
        "not wording; extra detail or different phrasing is fine. NO only if it is factually "
        "wrong, contradicts the reference, or misses the key point.\n"
        "2) VERBOSE — is it noticeably longer/more elaborate than a concise direct answer needs "
        "to be (a paragraph where a phrase would do)?\n"
        "Reply EXACTLY in this format:\nCORRECT: YES or NO\nVERBOSE: YES or NO\nREASON: <one line>"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You grade question-answering. Be lenient on wording "
             "and length when judging correctness; assess verbosity separately."},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        content = (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        return {"correct": False, "verbose": False, "reason": f"judge error: {e}"}

    def _flag(label: str) -> bool:
        m = re.search(rf"{label}\s*:\s*(YES|NO)", content, re.IGNORECASE)
        return bool(m) and m.group(1).upper() == "YES"

    m = re.search(r"REASON\s*:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
    reason = (m.group(1).strip() if m else content)[:140]
    return {"correct": _flag("CORRECT"), "verbose": _flag("VERBOSE"), "reason": reason}


def judge_correct(
    prediction: Any,
    references: Any,
    *,
    model: str = "minimax/minimax-m3",
    question: Optional[str] = None,
) -> "tuple[bool, str]":
    """Scorer wrapper around llm_judge -> (is_correct, detail with verbose flag)."""
    r = llm_judge(prediction, references, model=model, question=question)
    return r["correct"], f"correct={r['correct']} verbose={r['verbose']}: {r['reason']}"


def best_of(scorer: Scorer, pred: Any, answers: list) -> "tuple[bool, str]":
    """Score against a list of acceptable answers, keep the best."""
    best = (False, "")
    for a in answers:
        ok, detail = scorer(pred, a)
        if ok:
            return True, detail
        best = best if best[1] else (ok, detail)
    return best


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(
    name: str,
    examples: "list[Example]",
    scorer: Scorer,
    *,
    config: Optional[RLMConfig] = None,
    prefix: Optional[str] = None,
    verbose: bool = False,
    concurrency: int = 1,
    llm_kwargs: Optional[dict] = None,
) -> dict:
    """Run the RLM over `examples`, score each, and print a compact report.

    Samples run concurrently up to `concurrency` at a time (each `fast_rlm.run`
    is a blocking subprocess, so we offload it to a thread and gate with a
    semaphore). Results are reported as they complete; the summary preserves
    input order.
    """
    n = len(examples)
    concurrency = max(1, concurrency)
    print(f"\n=== {name}: {n} sample(s), concurrency={concurrency} ===\n")

    sem = asyncio.Semaphore(concurrency)

    async def run_one(i: int, ex: Example) -> dict:
        row = {"meta": ex.meta, "correct": False, "cost": 0.0,
               "total_tokens": 0, "input": 0, "cached": 0, "output": 0,
               "error": None, "detail": ""}
        async with sem:
            try:
                r = await asyncio.to_thread(
                    fast_rlm.run,
                    ex.query,
                    prefix=f"{prefix or name}_{i}",
                    config=config,
                    output_schema=ex.output_schema,
                    verbose=verbose,
                    llm_kwargs=llm_kwargs,
                )
                pred = r["results"]
                usage = r.get("usage") or {}
                row["cost"] = usage.get("cost") or 0.0
                row["total_tokens"] = usage.get("total_tokens") or 0
                # prompt_tokens already includes cached_tokens; split for display.
                row["cached"] = usage.get("cached_tokens") or 0
                row["input"] = (usage.get("prompt_tokens") or 0) - row["cached"]
                row["output"] = usage.get("completion_tokens") or 0
                # scorer receives the whole Example (so judges can use the
                # question/meta) and may be an LLM judge (network) → keep off loop.
                row["correct"], row["detail"] = await asyncio.to_thread(scorer, pred, ex)
            except Exception as e:  # one bad sample shouldn't kill the run
                row["error"] = str(e)
                row["detail"] = f"ERROR: {e}"

        tag = " ".join(f"{k}={v}" for k, v in ex.meta.items())
        mark = "✔" if row["correct"] else ("⚠" if row["error"] else "✗")
        print(f"[done {i + 1}/{n}] {tag}".rstrip())
        print(f"    {mark} {row['detail']}  "
              f"(${row['cost']:.4f}, {row['total_tokens']:,} tok)\n", flush=True)
        return row

    async def _gather() -> "list[dict]":
        return await asyncio.gather(*(run_one(i, ex) for i, ex in enumerate(examples)))

    rows = asyncio.run(_gather())
    n_correct = sum(r["correct"] for r in rows)
    n_err = sum(r["error"] is not None for r in rows)
    total_cost = sum(r["cost"] for r in rows)
    total_tok = sum(r["total_tokens"] for r in rows)

    model = (config.primary_agent if config is not None else None) or "default"
    print_results_table(name, model, rows)

    print(f"\n--- {name} summary ---")
    print(f"  model:        {model}")
    print(f"  accuracy:     {n_correct}/{n} = {n_correct / n:.1%}" if n else "  no samples")
    if n_err:
        print(f"  errors:       {n_err}/{n}")
    if n:
        print(f"  total cost:   ${total_cost:.4f}  (avg ${total_cost / n:.4f}/sample)")
        print(f"  total tokens: {total_tok:,}  (avg {total_tok // n:,}/sample)")

    return {
        "name": name,
        "model": model,
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "total_cost": total_cost,
        "total_tokens": total_tok,
        "rows": rows,
    }


def print_results_table(name: str, model: str, rows: "list[dict]") -> None:
    """Render a per-sample table with the token breakdown + a TOTAL row."""
    if not rows:
        return
    headers = ["#", "sample", "ok", "input", "cached-in", "output", "cost ($)"]

    def sample_label(r: dict) -> str:
        m = r.get("meta") or {}
        return ", ".join(f"{k}={v}" for k, v in m.items()) or "-"

    body = []
    for i, r in enumerate(rows, 1):
        ok = "⚠" if r["error"] else ("✔" if r["correct"] else "✗")
        body.append([
            str(i), sample_label(r), ok,
            f"{r['input']:,}", f"{r['cached']:,}", f"{r['output']:,}",
            f"{r['cost']:.4f}",
        ])
    total = [
        "", "TOTAL", f"{sum(r['correct'] for r in rows)}/{len(rows)}",
        f"{sum(r['input'] for r in rows):,}",
        f"{sum(r['cached'] for r in rows):,}",
        f"{sum(r['output'] for r in rows):,}",
        f"{sum(r['cost'] for r in rows):.4f}",
    ]

    table = [headers, *body, total]
    widths = [max(len(row[c]) for row in table) for c in range(len(headers))]

    def fmt(row, sep="  "):
        return sep.join(cell.ljust(widths[c]) for c, cell in enumerate(row))

    rule = "-" * (sum(widths) + 2 * (len(headers) - 1))
    print(f"\n=== {name} results — model: {model} ===")
    print(fmt(headers))
    print(rule)
    for r in body:
        print(fmt(r))
    print(rule)
    print(fmt(total))


def base_argparser(description: str) -> argparse.ArgumentParser:
    """Common CLI flags every benchmark script shares."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("-n", "--num-samples", type=int, default=3,
                   help="How many examples to evaluate (default: 3).")
    p.add_argument("--start", type=int, default=0,
                   help="Index into the dataset to start from (default: 0).")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p.add_argument("--only", default=None,
                   help="Comma-separated example indices (0-based) to keep after "
                        "sampling, e.g. '0,1'. Useful to re-run just the failures.")
    p.add_argument("-c", "--concurrency", type=int, default=4,
                   help="How many samples to run in parallel (default: 4).")
    p.add_argument("--verbose", action="store_true",
                   help="Stream the deno/engine output for each run (best with -c 1).")
    # Model selection — applies to both root and sub-agents unless overridden.
    p.add_argument("--model", default="minimax/minimax-m3",
                   help="Model for both root + sub agents (default: minimax/minimax-m3).")
    p.add_argument("--primary-agent", default=None, help="Override root-agent model.")
    p.add_argument("--sub-agent", default=None, help="Override sub-agent model.")
    # Budget caps (cumulative across root + all subagents). None -> RLMConfig default.
    p.add_argument("--max-prompt-tokens", type=int, default=None,
                   help="Cumulative prompt-token cap (default: RLMConfig's 200000).")
    p.add_argument("--max-completion-tokens", type=int, default=None,
                   help="Cumulative completion-token cap (default: RLMConfig's 50000).")
    p.add_argument("--max-money", type=float, default=None,
                   help="Hard USD budget cap (default: RLMConfig's 1.0).")
    # Extra LLM params spread into every chat.completions.create call.
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature for every LLM call (e.g. 0.3).")
    p.add_argument("--llm-kwargs", default=None,
                   help="JSON dict of extra LLM params, e.g. '{\"top_p\": 0.9}'. "
                        "Merged with --temperature.")
    # LLM-judge scoring (correct/not vs reference) instead of the default metric.
    p.add_argument("--judge", action="store_true",
                   help="Score with an LLM judge (correct/not) instead of the default metric.")
    p.add_argument("--judge-model", default="minimax/minimax-m3",
                   help="Model used for --judge scoring (default: minimax/minimax-m3).")
    # Ablation toggles (default ON; pass --no-* to disable that capability).
    p.add_argument("--no-tools", dest="enable_tools", action="store_false",
                   help="Disable user-defined tools (agent + prompt).")
    p.add_argument("--no-structured-io", dest="enable_structured_io",
                   action="store_false",
                   help="Disable output schemas / structured I/O (agent + prompt).")
    p.add_argument("--no-compression-guard", dest="enable_compression_guard",
                   action="store_false",
                   help="Disable the compression guard / batch judge / gather failsafe.")
    p.add_argument("--compression-ratio", type=float, default=None,
                   help="Trip the guard when a child gets >= this fraction of the parent context.")
    p.add_argument("--compression-min-chars", type=int, default=None,
                   help="Only guard contexts at least this many chars.")
    p.set_defaults(enable_tools=True, enable_structured_io=True,
                   enable_compression_guard=True)
    return p


def select_only(examples: "list[Example]", args) -> "list[Example]":
    """If --only was passed, keep just those (0-based, post-sampling) indices."""
    if not getattr(args, "only", None):
        return examples
    idxs = [int(x) for x in args.only.split(",") if x.strip() != ""]
    kept = [examples[i] for i in idxs if 0 <= i < len(examples)]
    print(f"(--only) keeping indices {idxs} -> {len(kept)} sample(s)")
    return kept


def llm_kwargs_from_args(args) -> Optional[dict]:
    """Build the llm_kwargs dict from --llm-kwargs (+ --temperature override)."""
    kw = {}
    if getattr(args, "llm_kwargs", None):
        kw.update(json.loads(args.llm_kwargs))
    if getattr(args, "temperature", None) is not None:
        kw["temperature"] = args.temperature
    return kw or None


def config_from_args(args) -> RLMConfig:
    """Build an RLMConfig from the shared CLI flags."""
    cfg = RLMConfig.default()
    cfg.primary_agent = args.primary_agent or args.model
    cfg.sub_agent = args.sub_agent or args.model
    cfg.enable_tools = args.enable_tools
    cfg.enable_structured_io = args.enable_structured_io
    cfg.enable_compression_guard = getattr(args, "enable_compression_guard", True)
    if getattr(args, "compression_ratio", None) is not None:
        cfg.compression_ratio = args.compression_ratio
    if getattr(args, "compression_min_chars", None) is not None:
        cfg.compression_min_chars = args.compression_min_chars
    if getattr(args, "max_prompt_tokens", None) is not None:
        cfg.max_prompt_tokens = args.max_prompt_tokens
    if getattr(args, "max_completion_tokens", None) is not None:
        cfg.max_completion_tokens = args.max_completion_tokens
    if getattr(args, "max_money", None) is not None:
        cfg.max_money_spent = args.max_money
    return cfg
