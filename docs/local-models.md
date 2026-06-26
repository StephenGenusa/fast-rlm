# Running RLMs on local models

Every option below — vLLM, llama.cpp, Ollama, LM Studio — exposes an
**OpenAI-compatible HTTP endpoint**, so a single client config drives all of
them through LiteLLM. What actually differs between them, *for RLMs
specifically*, is **concurrent throughput** — and that matters here more than for
a normal chatbot.

## Why throughput, not latency, is the RLM bottleneck

An RLM's whole speed strategy is `batch_llm_query(...)`: fire many sub-calls in
parallel, one per chunk, then reduce. A serial loop of sub-calls is the dominant
source of slowness. So the question for a local server is not "how fast is one
response" but "**how many sub-calls can it serve at once without queueing**." Set
the engine's `sub_max_parallel` to match the server's parallel capacity:

```python
RLM(client=root, sub_client=sub, sub_max_parallel=8)  # ≈ server's concurrent slots
```

If `sub_max_parallel` exceeds what the server can run concurrently, the extra
calls just queue and you lose the parallelism benefit.

## The four servers

### vLLM — best for high parallel throughput (your instinct is right)
Continuous batching + paged attention make vLLM the strongest choice for RLM's
many-parallel-sub-call pattern; it keeps the GPU busy across concurrent requests
instead of serializing them.

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 --max-num-seqs 16
```
```python
from rlm import RLM, LiteLLMClient, PyodideSandbox
root = LiteLLMClient("openai/Qwen/Qwen2.5-14B-Instruct",
                     api_base="http://localhost:8000/v1", api_key="sk-noop")
rlm  = RLM(client=root, sub_client=root, sub_max_parallel=12,
           sandbox=PyodideSandbox())
```

**The GGUF caveat.** vLLM's native strengths are safetensors / AWQ / GPTQ / FP8
weights, not GGUF. It *can* load GGUF, but support is secondary and typically
slower and less complete. So: if you want maximum parallel throughput and can use
HF/AWQ/GPTQ weights, vLLM wins. If you specifically need GGUF, prefer llama.cpp
or Ollama below. (You said you're newer to vLLM — the one-liner above is the
whole setup; the main knobs are `--max-num-seqs` for concurrency and
`--tensor-parallel-size` for multi-GPU.)

### llama.cpp — best for GGUF with real concurrency
`llama-server` serves GGUF and supports multiple parallel slots plus continuous
batching, so it gives you GGUF *and* meaningful sub-call parallelism on one box
(CPU, Metal, or a single GPU).

```bash
llama-server -m model.gguf --port 8080 --parallel 8 --cont-batching
```
```python
root = LiteLLMClient("openai/local", api_base="http://localhost:8080/v1",
                     api_key="sk-noop")
```
Match `sub_max_parallel` to `--parallel`.

### Ollama — easiest GGUF dev loop
Simplest to install and pull models; concurrency is capped lower. Set
`OLLAMA_NUM_PARALLEL` to allow concurrent requests.

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
ollama pull llama3.1
```
```python
root = LiteLLMClient("ollama/llama3.1", api_base="http://localhost:11434")
# or OpenAI-compat: LiteLLMClient("openai/llama3.1", api_base="http://localhost:11434/v1", api_key="ollama")
```

### LM Studio — GUI + OpenAI server
Convenient for experimentation; serves at `:1234/v1`. Concurrency is limited;
good for trying a model, less ideal as the parallel sub-call backend.

```python
root = LiteLLMClient("openai/your-loaded-model",
                     api_base="http://localhost:1234/v1", api_key="lm-studio")
```

## Picking root vs. sub models

The **root** model writes the code and orchestrates — give it a capable
instruct/coder model. The **sub** model does narrow extract/label/summarize work
over small chunks — a smaller, faster model is fine and much cheaper to run many
times in parallel. You can point them at two servers, or one server hosting both:

```python
rlm = RLM(
    client     = LiteLLMClient("openai/Qwen2.5-14B-Instruct", api_base="http://localhost:8000/v1", api_key="x"),
    sub_client = LiteLLMClient("openai/Qwen2.5-3B-Instruct",  api_base="http://localhost:8001/v1", api_key="x"),
    sub_max_parallel=12,
)
```

## OpenRouter (hosted, same client)

```python
LiteLLMClient("openrouter/anthropic/claude-3.5-sonnet")   # set OPENROUTER_API_KEY
LiteLLMClient("openrouter/qwen/qwen-2.5-72b-instruct")
```

## Quick recommendation

| You want | Use |
|---|---|
| Max parallel throughput, GPU, HF/AWQ/GPTQ weights ok | **vLLM** |
| GGUF + real concurrency on one box | **llama.cpp** (`--parallel`, `--cont-batching`) |
| Easiest GGUF dev loop | **Ollama** |
| GUI experimentation | **LM Studio** |
| No local GPU / hosted variety | **OpenRouter** |

Local serving has zero marginal cost, so LiteLLM's cost figure will read ~0 — the
`usage` token counts still populate for budgeting.
