"""Secret-vault decryption pipeline — showcases env_variables + tools + parallel subagents.

A bundle of encrypted intel messages is handed to the agent. To read any of
them it must call the `vault_decrypt` tool, which reads two credentials
(`VAULT_KEY`, `VAULT_PEPPER`) from `os.environ`. Those env vars are injected
**only inside Pyodide** via the `env_variables=` kwarg on `fast_rlm.run` — they
never appear in the model's context and are never set on the host process.

The agent is forced to fan out parallel subagents (each given the tool) to
decrypt batches in parallel, then aggregate into a categorized report.

Estimated time: ~1–2 min.
"""

import fast_rlm
from pydantic import BaseModel


VAULT_KEY = "spectre-7q"
VAULT_PEPPER = 11


def _encrypt(plaintext: str, key: str, pepper: int) -> str:
    """Host-side helper used to prepare the ciphertext bundle. NOT exposed to the agent."""
    return "".join(
        f"{((ord(c) + pepper) ^ ord(key[i % len(key)])):02x}"
        for i, c in enumerate(plaintext)
    )


# --- The single tool exposed to the agent's REPL -----------------------------

def vault_decrypt(ciphertext: str) -> str:
    """Decrypt a hex-encoded ciphertext from the intel vault.

    The vault's credentials are loaded from os.environ at call time:
      - VAULT_KEY    (str)  — the symmetric key
      - VAULT_PEPPER (int)  — additive pepper applied before XOR

    Raises KeyError if the credentials are not configured in the REPL.
    """
    import os
    key = os.environ["VAULT_KEY"]
    pepper = int(os.environ["VAULT_PEPPER"])
    data = bytes.fromhex(ciphertext)
    return "".join(
        chr((b ^ ord(key[i % len(key)])) - pepper) for i, b in enumerate(data)
    )


# --- The intel bundle --------------------------------------------------------

INTEL_MESSAGES = [
    "ALERT: perimeter breach detected on east gate sector 4",
    "INFO: weekly synthesis report archived under case 88-J",
    "LOG: routine sweep completed at 02:14 UTC, all stations green",
    "ALERT: unidentified drone hovering near antenna array",
    "INFO: courier package signed for by operative Echo-7",
    "LOG: generator B switched to backup mode for maintenance",
    "ALERT: encrypted radio chatter spiking on freq 412.6 MHz",
    "INFO: handoff scheduled with foreign liaison at 19:00 local",
    "LOG: server rack 11 cooling restored to nominal levels",
    "ALERT: badge cloning attempt logged at lobby reader 03",
]
CIPHERTEXTS = [_encrypt(m, VAULT_KEY, VAULT_PEPPER) for m in INTEL_MESSAGES]


# --- Output schema -----------------------------------------------------------

class CategorizedMessage(BaseModel):
    category: str   # "ALERT" | "INFO" | "LOG"
    plaintext: str


class Report(BaseModel):
    total: int
    alerts: int
    infos: int
    logs: int
    messages: list[CategorizedMessage]


structured_context = {
    "task": (
        "You have received a bundle of encrypted intel messages. Decrypt them, "
        "categorize each one by its leading tag (ALERT / INFO / LOG), and "
        "produce a final report."
    ),
    "ciphertexts": CIPHERTEXTS,
    "instructions": f"""
You MUST use parallel subagents to decrypt these in batches. Sequential is too slow.

You have one tool in your REPL: `vault_decrypt(ciphertext: str) -> str`.
It reads VAULT_KEY and VAULT_PEPPER from os.environ — those are already set
inside your REPL (they propagate to subagents automatically).

Strategy you MUST follow:

1. Split the ciphertexts list (length {len(CIPHERTEXTS)}) into EXACTLY 5 batches of 2.
2. Spawn 5 subagents in parallel via `asyncio.gather`. To each subagent:
     - pass its batch of 2 ciphertexts as the context (as a dict, e.g. {{"ciphertexts": [...]}})
     - pass the schema {{"type": "array", "items": {{"type": "object", "properties": {{"category": {{"type":"string"}}, "plaintext": {{"type":"string"}}}}, "required": ["category","plaintext"]}}}}
     - PASS THE TOOL EXPLICITLY: tools=[vault_decrypt]
   Each subagent decrypts its batch, splits each decrypted string into its
   leading category tag (the token before ":") and the rest as plaintext (no
   leading colon/space), and FINALs a JSON list of {{"category","plaintext"}} dicts.

3. After all 5 subagents return, flatten the lists and build:
     {{
       "total": <int>,
       "alerts": <count of category == "ALERT">,
       "infos":  <count of category == "INFO">,
       "logs":   <count of category == "LOG">,
       "messages": [ ... all CategorizedMessage entries, preserving original order ... ]
     }}
   FINAL that dict. The runner validates it against the Report schema.
""",
}


if __name__ == "__main__":
    config = fast_rlm.RLMConfig()
    config.primary_agent = "minimax/minimax-m2.5"
    config.sub_agent = "minimax/minimax-m2.5"
    config.max_depth = 2
    config.max_calls_per_subagent = 8
    config.max_money_spent = 0.30

    data = fast_rlm.run(
        structured_context,
        config=config,
        prefix="secret_vault",
        tools=[vault_decrypt],
        env_variables={"VAULT_KEY": VAULT_KEY, "VAULT_PEPPER": str(VAULT_PEPPER)},
        output_schema=Report,
    )

    print("\n=== DECRYPTED REPORT ===")
    r = data["results"]
    print(f"  total={r['total']}  alerts={r['alerts']}  infos={r['infos']}  logs={r['logs']}")
    for m in r["messages"]:
        tag = m["category"].ljust(6)
        print(f"  [{tag}] {m['plaintext']}")
    print("\nLOG:", data.get("log_file"))
    print("USAGE:", data.get("usage"))
