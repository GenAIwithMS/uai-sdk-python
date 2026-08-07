# Provider Development Kit (PDK)

> Stub — content to be written.

## Overview

The PDK provides tooling and templates to help contributors add new LLM provider support.

## Steps to Add a Provider

1. **Implement adapter**

   Create `src/uai/adapters/foo.py` and export it from
   `src/uai/adapters/__init__.py`:

   ```python
   from uai.adapters.base_adapter import BaseProviderAdapter

   class FooAdapter(BaseProviderAdapter):
       provider_name = "foo"

       def authenticate(self, credentials: dict[str, Any]) -> None:
           self._api_key = credentials.get("api_key")
           if not self._api_key:
               raise UAIAuthenticationError("Foo API key required")

       def format_request(self, request: UnifiedRequest) -> dict[str, Any]:
           ...  # UnifiedRequest -> provider wire format

       def parse_response(
           self, response: dict[str, Any], request: UnifiedRequest
       ) -> UnifiedResponse:
           ...  # provider response -> UnifiedResponse

       def handle_streaming(
           self, response: Any, request: UnifiedRequest
       ) -> Iterator[StreamChunk]:
           ...  # SSE parsing, TTFT, usage

       def translate_error(self, status_code: int, error_body: Any) -> Exception:
           ...  # provider errors -> UAIError subclasses

       def capabilities(self) -> dict[str, bool]:
           return {
               "chat": True,
               "streaming": True,
               "tools": True,
               "vision": False,
               "embeddings": False,
               "audio": False,
               "reasoning": False,
               "rerank": False,
               "tts": False,
               "transcription": False,
           }
   ```

   Adapters are **synchronous** — no `async` keywords.

2. **Register provider**

   Add a `ProviderConfig` to `src/uai/registry/providers.py` and insert it
   into `PROVIDER_REGISTRY`:
   ```python
   FOO_CONFIG = ProviderConfig(
       name="foo",
       display_name="Foo",
       base_url="https://api.foo.com/v1",
       auth_type=AuthType.BEARER_TOKEN,
       api_key_env_var="FOO_API_KEY",
       models={
           "foo-1": ProviderModel(...),
       },
       default_model="foo-1",
   )

   PROVIDER_REGISTRY["foo"] = FOO_CONFIG
   ```

3. **Write tests**
   - Unit: `tests/unit/test_adapters_foo.py` (mirror `test_adapters_kimi.py` /
     `test_adapters_stepfun.py` for the full auth / format / parse / streaming
     / error / capabilities coverage)
   - Integration: use local mock server in `tests/integration/`

4. **Document** in `docs/providers.md`.

## Validation

The PDK enforces adapter contracts through:
- Pydantic schema validation of `ProviderConfig`
- Capability matrix checks before every call (`FeatureNotSupportedError`)
- Adapter unit tests covering every contract method

## Versioning

The SDK targets all adapters against the shared `BaseProviderAdapter`
contract. When the contract changes, all existing adapters must be updated
in the same change so they stay in lock-step with the base class.