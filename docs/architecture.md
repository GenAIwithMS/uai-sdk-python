# Architecture

## Overview

The Universal AI Provider SDK is an infrastructure layer that unifies access to multiple LLM providers behind a single API.

### Three-Layer Architecture

```
Application
     │
     ▼
Universal AI SDK (client library)
     │  
     ▼
Provider Adapters (language-specific clients)
     │
  ┌──────┬────────┬─────────┬─────────┐
  ▼      ▼        ▼         ▼
DeepSeek  Qwen   GLM   Other Providers
(Kimi, StepFun, Doubao, MiniMax, Hunyuan)
```

## Key Concepts

### UnifiedRequest & UnifiedResponse

All requests are funneled through a `UnifiedRequest` object before hitting a provider adapter. All responses are normalized into a `UnifiedResponse`. This insulates application code from provider API schema changes.

### Provider Adapter Contract

Each provider has an adapter class implementing:
- `authenticate()`
- `format_request()` — map UnifiedRequest → provider schema
- `parse_response()` — map provider response → UnifiedResponse
- `handle_streaming()`
- `translate_error()`
- `capabilities()` — returns a boolean capability matrix

Adapters also expose non-chat feature hooks. Embeddings default to the shared
OpenAI-compatible `format_embed_request()` / `parse_embed_response()`, so most
providers inherit them as-is. Rerank is provider-specific; base
`format_rerank_request()` / `parse_rerank_response()` raise
`FeatureNotSupportedError` unless a provider overrides them (Qwen, GLM do).

### Capability Matrix

Providers advertise which features each model supports via
`ProviderCapabilities`, validated at import time and aggregated per provider
through `ProviderConfig.capabilities`. The **Capability Matrix Enforcer**
(Module 1.3.1) merges the registry's per-model capabilities with the active
adapter's `capabilities()` matrix — a feature is supported only when *both*
report `True`, so the SDK never fakes an implementation.

The client interrogates the enforcer at the top of every public method
(`chat`, `embed`, `rerank`) — including sub-features such as `tools`,
`streaming`, and image (`vision`) content — and raises
`FeatureNotSupportedError` instantly, before any middleware or network work:

```python
from uai import CapabilityMatrixEnforcer

enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-chat")
enforcer.supports("chat")    # True
enforcer.assert_supported("vision")  # raises FeatureNotSupportedError

client.supports("vision", provider="qwen", model="qwen-vl-max")  # True
```

The registry-level `check_capability(provider, model, capability)` helper
remains available for one-off assertions:

```python
from uai.registry import check_capability
check_capability("deepseek", "deepseek-chat", "vision")  # raises if unsupported
```

See [providers.md](providers.md) for the full matrix and [configuration.md](configuration.md)
for how configs are layered and overridden.

### UniversalAI Client

The `UniversalAI` class serves as the main orchestrator:
- Normalizes application requests into UnifiedRequest objects
- Routes requests to the appropriate provider adapter
- Handles streaming vs non-streaming responses
- Enforces capability checking before provider calls
- Manages authentication and API key distribution

The client can be instantiated with provider and model preferences:

```python
from uai import UniversalAI

client = UniversalAI(provider="deepseek", model="deepseek-chat")
response = client.chat(messages=[{"role": "user", "content": "Hello"}])
```

Optional middleware is registered explicitly via `client.use(...)` and wraps
requests with cross-cutting concerns (retry, circuit breaking, cache,
logging, tracing). An **Interceptor Execution Engine** (Module 1.4.1)
processes the chain: `before_request` hooks run in registration order, the
`execute` chain wraps the call, and `after_response`/`on_error` hooks run
in reverse. Middleware may also **halt the flow entirely** by raising
`MiddlewareHalt` with a response — skipping the network call while still
running `after_response`. The **CircuitBreakerMiddleware** (Module 1.4.2)
fast-fails a provider/model after repeated failures until it recovers, and
**RetryMiddleware** retries transient failures with exponential backoff
and jitter. See [middleware.md](middleware.md).

### Integration

Users interact with the SDK through the `UniversalAI.chat()` method:
- **Chat**: Send conversational messages to providers
- **Streaming**: Receive response chunks as they're generated
- **Tools**: Call provider-specific functions when supported
- **Structured Output**: Parse responses into structured data

Additional feature entry points route through the same adapter layer:
- **`client.embed()`**: text → embedding vectors (any embedding-capable model)
- **`client.rerank()`**: score documents against a query (Qwen, GLM)

The SDK maintains provider abstraction through:
1. Standardized `UnifiedRequest` and `UnifiedResponse` objects
2. Consistent error translation across providers
3. Capability enforcement based on provider model capabilities
4. Provider adapter encapsulation of provider-specific logic

### Performance Targets (Module 1.6)

The SDK commits to strict non-functional guarantees validated by an
automated KPI regression suite in `tests/performance/`, running against an
in-process **mock provider server** (`uai.testing.MockProviderServer`):

- Internal processing overhead < **5 ms** per request
- Streaming TTFT handling overhead < **30 ms**
- Throughput ≥ **1,000 requests/minute**
- Memory: idle < 50 MB marginal, < **150 MB** under ~100 parallel requests

Provider adapters are **lazy-loaded** on first use, keeping `import uai`
light for single-provider applications. See [performance.md](performance.md)
for how to run the suite and tune thresholds.