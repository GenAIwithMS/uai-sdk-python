# Architecture

> Stub — content to be written.

## Overview

The Universal AI Provider SDK is an infrastructure layer that unifies access to multiple LLM providers behind a single API.

### Three-Layer Architecture

```
Application
     │
     ▼
Universal AI SDK (client library)
     │  (optionally plugin/middleware pipeline)
     ▼
Provider Adapters (language-specific clients)
     │
  ┌──────┬────────┬─────────┐
  ▼      ▼        ▼         ▼
DeepSeek  Qwen   GLM   Other Providers...
```

## Key Concepts

### UnifiedRequest & UnifiedResponse

All requests are funneled through a `UnifiedRequest` object before hitting a provider adapter. All responses are normalized into a `UnifiedResponse`. This insulates application code from provider API schema changes.

### Middleware Pipeline

Advanced features (caching, retries, logging, routing) are implemented as opt-in middleware. By default, only core functionality runs.

```python
client.use(RetryMiddleware(max_retries=3))
client.use(CacheMiddleware(redis_client))
```

### Provider Adapter Contract

Each provider has an adapter class implementing:
- `authenticate()`
- `format_request()` — map UnifiedRequest → provider schema
- `parse_response()` — map provider response → UnifiedResponse
- `handle_streaming()`
- `translate_errors()`
- `capabilities()` — returns a boolean capability matrix

### Capability Matrix

Providers advertise which features each model supports via
`ProviderCapabilities`, validated at import time and aggregated per provider
through `ProviderConfig.capabilities`. Unsupported calls raise
`FeatureNotSupportedError` rather than silently failing.

```python
from uai.registry import check_capability

# Raises FeatureNotSupportedError if "vision" isn't supported for this model.
check_capability("deepseek", "deepseek-chat", "vision")
```

See [providers.md](providers.md) for the full matrix and [configuration.md](configuration.md)
for how configs are layered and overridden.