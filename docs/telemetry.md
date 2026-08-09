# Telemetry

The SDK emits standardized operational metrics using Prometheus naming
conventions (the `uai_*` namespace, Module 1.5.1) and per-call GenAI
tracing spans.

## Metrics (Prometheus-style)

Metrics are collected in-process by the **Metric Aggregation Engine**
(`MetricsRegistry`) and recorded automatically by `MetricsMiddleware`.
No OpenTelemetry dependency is required — the registry is dependency-free
and renderable in Prometheus text exposition format via `render()`.

| Metric | Type | Description |
|--------|------|-------------|
| `uai_requests_total` | Counter | Total SDK calls, tagged by operation/provider/model/status |
| `uai_provider_requests_total` | Counter | Per-provider calls, tagged by status |
| `uai_request_duration_seconds` | Histogram | End-to-end latency of the operation |
| `uai_ttft_seconds` | Histogram | Time-to-first-token (streaming) |
| `uai_tokens_input_total` | Counter | Prompt token usage |
| `uai_tokens_output_total` | Counter | Output token usage |
| `uai_cache_hits_total` | Counter | Cache hit count |
| `uai_retries_total` | Counter | Automatic retry count |
| `uai_errors_total` | Counter | Errors, tagged by operation/provider/model/type (exception class name) |

## Enabling metrics

```python
from uai import UniversalAI
from uai.middleware import MetricsMiddleware, MetricsRegistry

registry = MetricsRegistry()
client = UniversalAI(provider="deepseek")
client.use(MetricsMiddleware(registry=registry))

result = client.chat(messages=[{"role": "user", "content": "Hello"}])

# Inspect individual metrics
registry.counter_value("uai_requests_total", {"operation": "chat", "status": "success"})
registry.histogram_sum("uai_request_duration_seconds", {"operation": "chat"})

# Or render everything in Prometheus text format (e.g. for /metrics)
print(registry.render())
```

Share one `MetricsRegistry` across clients to aggregate globally. Register
`MetricsMiddleware` in the same chain as the other opt-in middleware
(`client.use(...)`).

## Tracing

Each LLM invocation generates a span annotated with GenAI semantic
attributes via `TracingMiddleware` (see [middleware.md](middleware.md),
Module 1.5.2):

| Attribute | Example |
|-----------|---------|
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | `deepseek` |
| `gen_ai.request.model` | `deepseek-chat` |
| `gen_ai.response.model` | `deepseek-chat` (may differ from requested) |
| `gen_ai.request.temperature` | `0.7` |
| `gen_ai.request.max_tokens` | `1024` |
| `gen_ai.response.id` | `cmpl-abc123` |
| `gen_ai.response.finish_reasons` | `["stop"]` |

Spans are recorded in-process by `SpanRecorder` (`recorder.spans`). With
`use_otel=True` (and the `opentelemetry` packages installed, or an
injected `tracer`), each invocation additionally creates a **discrete
distributed span** (kind `CLIENT`) on the `uai-sdk` tracer, so SDK calls
appear as their own nodes in a distributed trace.
