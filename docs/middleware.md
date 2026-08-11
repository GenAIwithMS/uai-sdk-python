# Middleware

Middleware are **opt-in** interceptors that wrap the client
request/response lifecycle. Nothing runs by default — register the pieces
you need with `client.use(...)`.

## Pipeline

The pipeline follows the interceptor pattern, executed by the
**`MiddlewareEngine`** (Module 1.4.1):

1. **`before_request`** hooks run in registration order — each may mutate
   the `UnifiedRequest`.
2. The request is executed through the **`execute` chain** — each
   middleware may wrap the next step (used by retry and cache).
3. **`after_response`** hooks run in reverse registration order — each may
   mutate the response.
4. **`on_error`** hooks run (in reverse order) when the chain raises.

Middleware are **synchronous**, matching the synchronous client.

The engine (`MiddlewareEngine`) is normally used through the client — it
owns one and delegates `use()` and the pipeline runs to it. It is also
exported so advanced users can compose and test pipelines standalone.

## Halting the flow (MiddlewareHalt)

Any middleware can **halt the execution flow entirely** based on runtime
conditions (Module 1.4.1) by raising `MiddlewareHalt` with a response. The
engine short-circuits — the execute chain and the network call are
skipped, and the supplied response is fed through the `after_response`
hooks. `on_error` is **not** invoked: a halt is a deliberate outcome, not
an error.

```python
from uai.middleware import MiddlewareHalt
from uai.models import UnifiedResponse
from uai.middleware.base import BaseMiddleware

class FallbackMiddleware(BaseMiddleware):
    """Serve a canned response for a blocked provider without any network call."""

    def before_request(self, request, context):
        if context.provider == "deepseek":
            raise MiddlewareHalt(
                UnifiedResponse(content="fallback: deepseek unavailable", model=context.model)
            )
        return request

client.use(FallbackMiddleware())
result = client.chat(messages=[{"role": "user", "content": "Hi"}])
print(result.content)  # "fallback: deepseek unavailable"
```

Halting works from any hook (`before_request`, the `execute` chain) and
for both non-streaming and streaming calls (a streaming halt replaces the
provider stream with an iterable of chunks). Use it for guardrails,
fallbacks, rate-limiting without burning tokens, or feature flags.

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
| `CircuitBreakerMiddleware` | Fast-fails a degraded provider/model until it recovers (Module 1.4.2) |
| `CacheMiddleware` | In-memory TTL cache for identical non-streaming requests |
| `LoggingMiddleware` | Structured request/response/error log lines, correlated by `request_id` |
| `TracingMiddleware` | Records one span per call with GenAI semantic attributes (in-process recorder, optional OpenTelemetry export) |
| `MetricsMiddleware` | Records Prometheus-style metrics (requests, latency, TTFT, tokens, cache, retries, errors) into a `MetricsRegistry` |

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

### CircuitBreakerMiddleware

Resiliency complement to retry (Module 1.4.2): detect sustained failures
for a provider/model pair and **fast-fail** subsequent requests until the
provider recovers, avoiding wasted network calls and throttling. The
circuit has three states per (provider, model) key:

- **closed** — normal operation; failures are counted
- **open** — requests are rejected immediately, no network call
- **half_open** — after `reset_timeout`, one trial request (probe) is
  allowed; success closes the circuit, failure reopens it

```python
from uai.middleware import CircuitBreakerMiddleware, RetryMiddleware

client.use(CircuitBreakerMiddleware(
    failure_threshold=5,   # consecutive failures before the circuit opens
    reset_timeout=30.0,    # seconds before a half-open probe is allowed
    fallback_response=None,  # optional response served while open
))
client.use(RetryMiddleware(max_retries=3))
```

Register the breaker **before** (outside of) `RetryMiddleware` so an open
circuit short-circuits immediately instead of consuming retries.

While the circuit is open, requests raise `UAICircuitOpenError` — unless
`fallback_response` is set, in which case that response is served via
`MiddlewareHalt` (no error, `after_response` still runs). State can be
inspected and managed programmatically:

```python
breaker = CircuitBreakerMiddleware(failure_threshold=5)
client.use(breaker)

breaker.state("deepseek", "deepseek-v4-flash")  # "closed" | "open" | "half_open"
breaker.failures("deepseek", "deepseek-v4-flash")
breaker.reset()                              # close everything
```

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

Each call produces a `Span` with OpenTelemetry-style GenAI attributes
(Module 1.5.2):

- `gen_ai.operation.name`, `gen_ai.provider.name`,
  `gen_ai.request.model`, `gen_ai.request.temperature`,
  `gen_ai.request.top_p`, `gen_ai.request.max_tokens`,
  `gen_ai.request.stop`, `gen_ai.request.tools`
- `gen_ai.response.model` (the model that actually served the response),
  `gen_ai.response.id`, `gen_ai.response.finish_reasons`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.usage.cache_read_input_tokens`,
  `gen_ai.usage.cache_creation_input_tokens`

If the `opentelemetry` packages are installed, pass `use_otel=True` and
each invocation additionally creates a **discrete distributed span**
(kind `CLIENT`) on the `uai-sdk` tracer — not just attributes on the
current span — so every SDK call appears as its own node in your trace.
A custom `tracer` can be injected instead of the installed packages.

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
