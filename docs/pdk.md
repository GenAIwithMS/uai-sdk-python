# Provider Development Kit (PDK)

> Stub — content to be written.

## Overview

The PDK provides tooling and templates to help contributors add new LLM provider support.

## Steps to Add a Provider

1. **Implement adapter**

   ```python
   from uai.adapters.base import BaseProviderAdapter

   class FooAdapter(BaseProviderAdapter):
       async def authenticate(self): ...
       async def format_request(self, request): ...
       async def parse_response(self, raw): ...
       async def handle_streaming(self, raw_stream): ...
       async def translate_errors(self, exc): ...
       def capabilities(self) -> ProviderCapabilities: ...
   ```

2. **Register provider**

   Add to `src/uai/registry/providers.py`:
   ```python
   PROVIDER_REGISTRY["foo"] = ProviderConfig(
       name="foo",
       base_url="https://api.foo.com/v1",
       auth_type=AuthType.BEARER_TOKEN,
       models=[...],
       ...
   )
   ```

3. **Write tests**
   - Unit: `tests/unit/test_adapters.py`
   - Integration: use local mock server in `tests/integration/`

4. **Document** in `docs/providers.md`.

## Validation

The PDK enforces adapter contracts through:
- Pydantic schema validation of `ProviderConfig`
- Capability matrix checks before every call
- Automated contract tests (mock server)

## Versioning

Adapters declare a version and required API version. The SDK warns if an adapter is outdated.