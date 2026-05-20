"""Multi-symbol market scanner — showcases user-defined TOOLS and parallel subagents.

The agent is given three Python tools (`fetch_ohlc`, `sma`, `rsi`) that it can
call directly inside its Pyodide REPL — no source is shown, only their
signatures and docstrings. The prompt requires the agent to fan out 5 parallel
subagent calls (one per symbol), explicitly handing the tools down to each
child so they can run the same analysis. The root then aggregates the verdicts.

Estimated time: ~1–2 min.
"""

import fast_rlm
from pydantic import BaseModel


# --- Tools exposed to the agent's REPL ---------------------------------------

def fetch_ohlc(symbol: str, days: int = 60) -> list:
    """Fetch deterministic pseudo-OHLC candles for a symbol.

    Returns a list of dicts: [{"open","high","low","close","volume"}, ...] of
    length `days`. Deterministic given the symbol (seeded via SHA-256).
    """
    import hashlib
    h = hashlib.sha256(symbol.encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    price = 50.0 + (seed % 450)
    candles = []
    for _ in range(days):
        h = hashlib.sha256(h).digest()
        drift = (int.from_bytes(h[:4], "big") % 2000 - 1000) / 1000.0  # ~U(-1,+1)
        op = price
        cl = max(0.5, op * (1 + drift * 0.04))
        hi = max(op, cl) * (1 + (int.from_bytes(h[4:6], "big") % 200) / 5000.0)
        lo = min(op, cl) * (1 - (int.from_bytes(h[6:8], "big") % 200) / 5000.0)
        vol = int.from_bytes(h[8:12], "big") % 1_000_000
        candles.append({
            "open": round(op, 2), "high": round(hi, 2),
            "low": round(lo, 2), "close": round(cl, 2), "volume": vol,
        })
        price = cl
    return candles


def sma(values: list, n: int) -> float:
    """Simple moving average over the trailing n values. Returns 0.0 if empty."""
    if not values or n <= 0:
        return 0.0
    window = values[-n:]
    return round(sum(window) / len(window), 4)


def rsi(closes: list, period: int = 14) -> float:
    """Wilder-style RSI over the trailing `period` closes. Returns 50.0 if data is too short."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return round(100 - (100 / (1 + rs)), 2)


# --- Output schema -----------------------------------------------------------

class SymbolVerdict(BaseModel):
    symbol: str
    last_close: float
    sma_5: float
    sma_20: float
    rsi: float
    verdict: str   # "bullish" | "bearish" | "neutral"


SYMBOLS = ["ACME", "BLOB", "CORP", "DYNE", "EONX"]


prompt = f"""
You are a market scanner. Analyze the following symbols IN PARALLEL using
subagents (this is REQUIRED — sequential analysis is too slow):

    SYMBOLS = {SYMBOLS}

You have three tools pre-loaded in your REPL: `fetch_ohlc`, `sma`, `rsi`.
Their signatures and docstrings are shown in the initial probe.

Strategy you MUST follow:

1. Spawn EXACTLY {len(SYMBOLS)} subagents using `asyncio.gather` — one per symbol.
2. **Each subagent needs the tools**, but sub-agents do NOT inherit them by
   default. You MUST pass them explicitly:

        await llm_query(task_for_symbol, schema, tools=[fetch_ohlc, sma, rsi])

3. Each subagent should:
     - Call `fetch_ohlc(symbol, days=60)` to get candles
     - Extract the list of close prices
     - Compute `sma(closes, 5)`, `sma(closes, 20)`, and `rsi(closes, 14)`
     - Decide a verdict:
         * "bullish"  if sma_5 > sma_20 and rsi < 70
         * "bearish"  if sma_5 < sma_20 and rsi > 30
         * "neutral"  otherwise
     - FINAL a dict matching this JSON schema:
         {{
             "symbol": str, "last_close": float,
             "sma_5": float, "sma_20": float,
             "rsi": float, "verdict": str
         }}
       You should pass that schema as the second positional arg of llm_query.

4. After all subagents return, sort the list by `rsi` DESCENDING and FINAL it.
   The root output schema is a JSON array matching `list[SymbolVerdict]`.
"""


if __name__ == "__main__":
    config = fast_rlm.RLMConfig()
    config.primary_agent = "minimax/minimax-m2.5"
    config.sub_agent = "minimax/minimax-m2.5"
    config.max_depth = 2
    config.max_calls_per_subagent = 8
    config.max_money_spent = 0.30

    data = fast_rlm.run(
        prompt,
        config=config,
        prefix="market_scanner",
        tools=[fetch_ohlc, sma, rsi],
        output_schema=list[SymbolVerdict],
    )

    print("\n=== MARKET SCAN ===")
    for row in data["results"]:
        verdict = row["verdict"].upper().ljust(8)
        print(f"  {row['symbol']:<5}  close={row['last_close']:>8.2f}  "
              f"SMA5={row['sma_5']:>8.2f}  SMA20={row['sma_20']:>8.2f}  "
              f"RSI={row['rsi']:>6.2f}  {verdict}")
    print("\nLOG:", data.get("log_file"))
    print("USAGE:", data.get("usage"))
