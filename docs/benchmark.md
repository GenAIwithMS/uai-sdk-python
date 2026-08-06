# Benchmark CLI

> Stub — Phase 2 content.

## Overview

The `uai` CLI provides an offline benchmarking tool to compare provider performance.

## Usage

```bash
# Compare two providers on a fixed prompt
uai benchmark \
  --providers deepseek,qwen \
  --prompt "Explain the theory of relativity" \
  --iterations 10 \
  --parallel 2

# Output includes TTFT, latency, tokens/sec, estimated cost
```

## Metrics Reported

| Metric | Description |
|--------|-------------|
| TTFT (ms) | Time-to-first-token |
| Latency (ms) | Full response time |
| Tokens/sec | Throughput |
| Est. cost ($) | Approximate provider cost |
| Error rate (%) | Failed requests