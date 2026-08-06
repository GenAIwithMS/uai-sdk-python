# Middleware

> Stub — content to be written.

## Overview

Middleware are opt-in interceptors that wrap the request/response lifecycle. They follow the **decorator/interceptor pattern** and implement `before_request()` and `after_response()` hooks.

## Built-in Middleware (MVP)

| Middleware | Phase | Description |
|------------|-------|-------------|
| `LoggingMiddleware` | MVP | Logs request metadata and errors |
| `RetryMiddleware` | MVP | Reties transient failures with exponential backoff + jitter |
| `CacheMiddleware` | Phase 2 | In-memory or Redis-backed response caching |
| `TracingMiddleware` | Phase 2 | OpenTelemetry span injection |
| `RedactionMiddleware` | Phase 3 | PII redaction before egress |
| `RouterMiddleware` | Phase 3 | Provider selection by strategy (fallback, cost, latency) |

## Writing Custom Middleware

```python
from uai.middleware.base import BaseMiddleware

class MyCustomMiddleware(BaseMiddleware):
    async def before_request(self, request, context):
        # mutate request, check cache, enforce rate limits
        ...
        return request

    async def after_response(self, response, context):
        # log, cache, validate output
        ...
        return response

# Register:
client.use(MyCustomMiddleware())
```