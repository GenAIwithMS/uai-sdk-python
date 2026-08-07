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
through `ProviderConfig.capabilities`. Unsupported calls raise
`FeatureNotSupportedError` rather than silently failing.

```python
from uai.registry import check_capability

# Raises FeatureNotSupportedError if "vision" isn't supported for this model.
check_capability("deepseek", "deepseek-chat", "vision")
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