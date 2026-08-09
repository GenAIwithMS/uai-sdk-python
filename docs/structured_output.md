# Structured Output

The SDK can enforce structured JSON output from LLM responses by validating the output against a Pydantic model.

## How It Works

1. `output_schema` accepts a Pydantic model class (`type[BaseModel]`) and is
   stored on the `UnifiedRequest` as `output_schema`.
2. The request is sent to the provider normally (no schema is injected into
   the request).
3. When the provider's response is parsed by the provider adapter
   (`parse_response`), the SDK validates the returned JSON against the
   schema:
   - The content is parsed as JSON (`json.loads`).
   - The parsed payload is validated with `schema.model_validate(...)`.
4. On malformed JSON or validation failure, a `ResponseParsingError` is
   raised with the provider name attached.
5. On success, the validated model is exposed on the response's `parsed`
   field.

## Usage (adapter level)

Structured-output validation is implemented in the provider adapter layer
(module 1.2) — both `DeepSeekAdapter` and `QwenAdapter` implement it in
`parse_response()`:

```python
from pydantic import BaseModel

from uai.adapters.deepseek import DeepSeekAdapter
from uai.models import UnifiedRequest

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

adapter = DeepSeekAdapter()
request = UnifiedRequest(
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
    output_schema=Summary,
)

# raw_provider_response = {json from POST {base_url}/chat/completions}
result = adapter.parse_response(raw_provider_response, request)
print(result.parsed.title)
```

## Client wiring (pending)

`client.chat()` accepts `output_schema` and stores it on the
`UnifiedRequest`, but its inline chat response parser does not perform the
validation yet — `result.parsed` is currently `None` when going through the
client. Once `output_schema` is wired end-to-end, the client example will
be:

```python
from uai import UniversalAI

client = UniversalAI(api_key="sk-...", provider="deepseek")
result = client.chat(
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
    output_schema=Summary,
)
print(result.parsed.title)
```

## Errors

Validation failures raise `ResponseParsingError` with the provider name
attached:

```python
from uai.exceptions import ResponseParsingError

try:
    result = adapter.parse_response(raw_provider_response, request)
except ResponseParsingError as exc:
    print(f"Schema validation failed: {exc}")
```
