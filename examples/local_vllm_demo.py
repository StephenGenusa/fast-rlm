"""
Run an RLM entirely on local models, sandboxed.

Serve any OpenAI-compatible endpoint first, e.g.:
    vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 --max-num-seqs 16
    # or: llama-server -m model.gguf --port 8000 --parallel 8 --cont-batching
    # or: ollama serve  (OpenAI-compat at :11434/v1)

Then (needs Deno or Node + `npm install pyodide` for the sandbox):
    python examples/local_vllm_demo.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlm import RLM, LiteLLMClient, PyodideSandbox  # noqa: E402

API_BASE = os.getenv("RLM_API_BASE", "http://localhost:8000/v1")
MODEL = os.getenv("RLM_MODEL", "openai/Qwen/Qwen2.5-14B-Instruct")


def main():
    client = LiteLLMClient(MODEL, api_base=API_BASE, api_key="sk-noop")
    rlm = RLM(
        client=client,
        sub_client=client,                 # or a smaller model on another port
        sandbox=PyodideSandbox(runtime="auto"),  # prefers Deno
        sub_max_parallel=8,                # match your server's parallel slots
        max_depth=1,
        verbose=True,
    )

    # A context the model must explore programmatically.
    rows = [f"Row {i} || score={i % 7}" for i in range(3000)]
    context = "\n".join(rows)

    result = rlm.complete(
        query="How many rows have score == 3? Answer with just the number.",
        context=context,
    )
    print("\n--- RESULT ---")
    print("answer:", result.answer)
    print("usage :", result.usage)


if __name__ == "__main__":
    main()
