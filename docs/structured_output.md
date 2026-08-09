# Structured Output

The SDK can enforce structured JSON output from LLM responses by validating
the output against a Pydantic model (Module 1.3.2).

## How It Works

1. `output_schema` accepts a Pydantic model class (`type[BaseModel]`) and is
   stored on the `UnifiedRequest` as `output_schema`.
2. The request is sent to the provider with the schema's JSON Schema
   injected as a **system-prompt instruction**, nudging the model toward a
   schema-conforming JSON object. (No provider-specific
   `response_format` parameter is used — injection works with every
   OpenAI-compatible provider.)
3. When the response arrives, the SDK parses and validates the returned
   content against the schema:
   - The content is parsed as JSON — markdown code fences and leading prose
     are tolerated (`extract_json_object`).
   - The parsed payload is validated with `schema.model_validate(...)`.
4. On malformed JSON or validation failure, a `ResponseParsingError` is
   raised with the provider name attached.
5. On success, the validated model is exposed on the response's `parsed`
   field — on `UnifiedResponse.parsed` for non-streaming calls, and on the
   final `StreamChunk.parsed` for streaming calls.

## Usage (client)

```python
from pydantic import BaseModel

from uai import UniversalAI

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

client = UniversalAI(api_key="sk-...", provider="deepseek")
result = client.chat(
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
    output_schema=Summary,
)
print(result.parsed.title)
```

### Streaming

`output_schema` also works with `stream=True`. The SDK accumulates content
deltas across the stream and validates the assembled payload when the stream
ends; the validated model is attached to the final chunk:

```python
chunks = client.chat(messages=[...], output_schema=Summary, stream=True)
parsed = None
for chunk in chunks:
    if chunk.parsed is not None:
        parsed = chunk.parsed
print(parsed.title)
```

If the assembled payload is malformed JSON or violates the schema,
`ResponseParsingError` is raised when the stream ends (on the final chunk) —
same as the non-streaming path.

## Usage (adapter level)

The same validation is available directly on every provider adapter
(`parse_response`), shared via `uai.structured.parse_structured_output`:

```python
from uai.adapters.deepseek import DeepSeekAdapter
from uai.models import UnifiedRequest

adapter = DeepSeekAdapter()
request = UnifiedRequest(
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
    output_schema=Summary,
)
result = adapter.parse_response(raw_provider_response, request)
print(result.parsed.title)
```

## Retrying validation failures

`ResponseParsingError` is designed to be caught and automatically retried
by the middleware pipeline. RetryMiddleware retries it only when explicitly
enabled (`retry_on_parsing_error=True`) — the default is off so existing
users don't incur surprise retries:

```python
from uai import UniversalAI
from uai.middleware import RetryMiddleware

client = UniversalAI(api_key="sk-...", provider="deepseek")
client.use(RetryMiddleware(max_retries=2, retry_on_parsing_error=True))
```

## Errors

Validation failures raise `ResponseParsingError` with the provider name
attached:

```python
from uai.exceptions import ResponseParsingError

try:
    result = client.chat(messages=[...], output_schema=Summary)
except ResponseParsingError as exc:
    print(f"Schema validation failed: {exc}")
```
