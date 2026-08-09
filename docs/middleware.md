# Middleware

Middleware are **opt-in** interceptors that wrap the client
request/response lifecycle. Nothing runs by default — register the pieces
you need with `client.use(...)`.

## Pipeline

The pipeline follows the interceptor pattern:

1. **`before_request`** hooks run in registration order — each may mutate
   the `UnifiedRequest`.
2. The request is executed through the **`execute` chain** — each
   middleware may wrap the next step (used by retry and cache).
3. **`after_response`** hooks run in reverse registration order — each may
   mutate the response.
4. **`on_error`** hooks run (in reverse order) when the chain raises.

Middleware are **synchronous**, matching the synchronous client.

```python
from uai import UniversalAI
from uai.middleware import CacheMiddleware, LoggingMiddleware, RetryMiddleware

client = UniversalAI(provider="deepseek")
client.use(LoggingMiddleware())
client.use(RetryMiddleware(max_retries=3))
client.use(CacheMiddleware(ttl=300))
```

`use()` accepts a single middleware or a list and returns the client, so
calls can be chained.

## Built-in Middleware

| Middleware | Description |
|------------|-------------|
| `RetryMiddleware` | Retries transient failures (429, 5xx, network, timeout) with exponential backoff + jitter |
| `CacheMiddleware` | In-memory TTL cache for identical non-streaming requests |
| `LoggingMiddleware` | Structured request/response/error log lines, correlated by `request_id` |
| `TracingMiddleware` | Records one span per call with GenAI semantic attributes (in-process recorder, optional OpenTelemetry export) |

### RetryMiddleware

```python
from uai.middleware import RetryMiddleware

client.use(RetryMiddleware(
    max_retries=3,     # retry attempts after the first call
    base_delay=0.5,    # initial backoff (seconds)
    max_delay=10.0,    # backoff cap
    jitter=True,       # randomize delays
))
```

Retryable failures: `UAIRateLimitError` (429, honors `Retry-After` when
present), `UAINetworkError`, `UAITimeoutError`, and 5xx server errors.
Authentication and other 4xx errors are **not** retried.

Structured-output validation failures (`ResponseParsingError`, Module
1.3.2) are retried only when explicitly enabled:

```python
client.use(RetryMiddleware(max_retries=2, retry_on_parsing_error=True))
```

For streaming, retries only happen if the failure occurs before the first
chunk is delivered; once streaming has started, errors propagate as-is.

### CacheMiddleware

```python
from uai.middleware import CacheMiddleware

client.use(CacheMiddleware(ttl=300, max_size=1024))
```

Responses are keyed by a hash of the normalized `UnifiedRequest`, so
identical calls are served from memory instead of hitting the provider
(billing). Streaming requests and embed/rerank calls are **not** cached.

### LoggingMiddleware

```python
from uai.middleware import LoggingMiddleware

client.use(LoggingMiddleware())
```

Logs one line per request and per response (`operation`, `provider`,
`model`, latency, usage, `finish_reason`, `request_id`) and one warning per
final error. Secrets are never logged.

### TracingMiddleware

```python
from uai.middleware import SpanRecorder, TracingMiddleware

recorder = SpanRecorder()
client.use(TracingMiddleware(recorder=recorder, service_name="my-app"))

# ... after requests ...
for span in recorder.spans:
    print(span.operation, span.duration_ms, span.status)
```

Each call produces a `Span` with OpenTelemetry-style GenAI attributes:

- `gen_ai.operation.name`, `gen_ai.request.model`,
  `gen_ai.request.temperature`, `gen_ai.request.max_tokens`
- `gen_ai.response.model`, `gen_ai.response.finish_reasons`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`

If the `opentelemetry` packages are installed, pass `use_otel=True` to also
export the attributes onto the current OpenTelemetry span.

## Writing Custom Middleware

```python
from uai.middleware.base import BaseMiddleware

class MyMiddleware(BaseMiddleware):
    name = "my-middleware"

    def before_request(self, request, context):
        # mutate request, enforce rate limits, redact payloads ...
        return request

    def after_response(self, response, context):
        # log, validate output, record metrics ...
        return response

    def on_error(self, error, context):
        # called when the chain raises ...

client.use(MyMiddleware())
```

The `execute` hook lets a middleware wrap the actual call (used by retry
and cache) — see `BaseMiddleware` in `uai.middleware.base`.
