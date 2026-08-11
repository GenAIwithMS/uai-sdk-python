# Benchmark CLI

The `uai` CLI provides an offline benchmarking tool that measures every
**chat-capable model** in the provider registry (DeepSeek, Qwen, GLM,
Kimi, StepFun, Doubao, MiniMax, Hunyuan). Embedding/rerank-only models are
skipped automatically.

## Usage

```bash
# Benchmark every chat model of every provider with an API key
uai benchmark

# Limit to specific providers / models
uai benchmark --providers deepseek,qwen
uai benchmark --providers qwen --models qwen3.7-plus,qwen-vl-max

# Tune the workload
uai benchmark \
  --prompt "Explain the theory of relativity" \
  --iterations 10 \
  --parallel 4 \
  --max-tokens 256

# Non-streaming (no TTFT column; uses provider-reported usage)
uai benchmark --no-stream

# Machine-readable output
uai benchmark --json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--providers` | all | Comma-separated provider names |
| `--models` | all chat models | Comma-separated model ids |
| `--prompt` | fixed default | Prompt text to send |
| `--iterations` | 10 | Calls per model |
| `--parallel` | 4 | Parallel workers |
| `--max-tokens` | 256 | Max output tokens per call |
| `--temperature` | 0.0 | Sampling temperature |
| `--no-stream` | — | Disable streaming (no TTFT) |
| `--json` | — | Emit results as JSON |

Providers whose API key env var is not set are skipped with a note on
stderr; if none are configured, the command exits with code 2.

## Metrics Reported

| Metric | Description |
|--------|-------------|
| TTFT ms | Time-to-first-token (streaming calls, first content chunk) |
| Latency ms | Full round-trip time |
| Tok/s | Output tokens per second |
| Cost $ | Estimated cost from provider pricing and token usage |
| Err % | Failed requests / iterations |

## Programmatic API

```python
from uai.benchmark import benchmark_models

results = benchmark_models(providers=["deepseek"], iterations=5)

for r in results:
    print(
        r.provider, r.model,
        f"avg latency: {r.latency_ms_avg:.0f}ms",
        f"TTFT: {r.ttft_ms_avg:.0f}ms",
        f"err: {r.error_rate:.0%}",
        f"cost: ${r.cost_usd_total:.6f}",
    )
```

`benchmark_models()` returns one `BenchmarkResult` per model with
`SampleResult`s for each iteration (`latency_ms`, `ttft_ms`, token usage,
estimated cost, error). Runs are parallelized with a thread pool.
