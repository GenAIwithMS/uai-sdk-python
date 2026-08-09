# Telemetry

> Stub — content to be written.

## Overview

The SDK emits standardized metrics and traces using OpenTelemetry conventions for Generative AI.

## Metrics (Prometheus-style)

| Metric | Type | Description |
|--------|------|-------------|
| `uai_requests_total` | Counter | Total SDK calls, tagged by provider/status |
| `uai_request_duration_seconds` | Histogram | End-to-end latency |
| `uai_ttft_seconds` | Histogram | Time-to-first-token (streaming) |
| `uai_tokens_input_total` | Counter | Prompt token usage |
| `uai_tokens_output_total` | Counter | Output token usage |
| `uai_cache_hits_total` | Counter | Cache hit count |
| `uai_retries_total` | Counter | Automatic retry count |

## Tracing

Each LLM invocation generates a span annotated with GenAI semantic attributes:

| Attribute | Example |
|-----------|---------|
| `gen_ai.operation.name` | `chat` |
| `gen_ai.request.model` | `deepseek-chat` |
| `gen_ai.response.model` | `deepseek-chat` |
| `gen_ai.request.temperature` | `0.7` |
| `gen_ai.request.max_tokens` | `1024` |
| `gen_ai.response.finish_reasons` | `["stop"]` |

## Enabling Telemetry

> **Note:** Telemetry *metrics* are a roadmap item — no OpenTelemetry
> metric/export wiring exists in the SDK yet. Tracing per-call spans is
> available today via `TracingMiddleware` (see [middleware.md](middleware.md));
> the example below shows the intended opt-in surface. The client
> constructor accepts only `api_key`, `provider`, `model`, `credentials`,
> `timeout`, and `max_retries`.

```python
from uai import UniversalAI

client = UniversalAI(
    api_key="...",
    provider="deepseek",
    # enable_telemetry=True  # planned (not yet implemented)
)
```