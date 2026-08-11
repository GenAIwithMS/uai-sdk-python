# Chat

Text-in/text-out conversational completions. Requests are normalized into a
`UnifiedRequest` and sent to the provider's chat completions endpoint.

## Basic Usage

```python
from uai import UniversalAI

client = UniversalAI(api_key="sk-...", provider="deepseek")

result = client.chat(
    messages=[{"role": "user", "content": "Explain quantum computing in 2 sentences."}],
    model="deepseek-v4-flash",
)
print(result.content)
```

## With Message History

```python
result = client.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather today?"},
    ],
    model="qwen-turbo",
)
```

Messages are passed as a list of dicts (or `uai.models.ChatMessage`
instances). There is no `prompt` convenience parameter — always use
`messages`.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list[dict]` / `list[ChatMessage]` | Conversation history (role/content, optional tool_calls, tool_call_id) |
| `provider` | `str` | Target provider (defaults to the client's provider) |
| `model` | `str` | Model name (defaults to the provider's default model) |
| `max_tokens` | `int` | Max output tokens |
| `temperature` | `float` | Sampling temperature (0.0–2.0) |
| `top_p` | `float` | Top-p nucleus sampling (0.0–1.0) |
| `stop` | `list[str]` / `str` | Stop sequence(s) where generation should halt |
| `stream` | `bool` | Enable streaming; returns an iterator of `StreamChunk` |
| `stream_callback` | `Callable[[StreamChunk], None]` | Optional callback invoked for every streamed chunk |
| `tools` | `list[dict]` / `list[ToolDefinition]` | Function/tool definitions |
| `output_schema` | `type[BaseModel]` | Pydantic model for structured output validation; the schema is injected as a system-prompt instruction and the response is validated against it (`result.parsed`) |

Additional `UnifiedRequest` fields (`frequency_penalty`, `presence_penalty`,
`tool_choice`, `user`, `metadata`) can be passed as keyword arguments.

## Capability enforcement

`chat()` validates the capability matrix (Module 1.3.1) before any network
or middleware work and raises `FeatureNotSupportedError` immediately if a
requested feature is unsupported for the resolved model:

| Requested feature | Capability checked |
|-------------------|--------------------|
| `chat()` call itself | `chat` |
| `tools=[...]` | `tools` |
| `stream=True` | `streaming` |
| any message with an `ImageContent` block | `vision` |

For example, passing tool definitions to a model without tool support, or
image content to a text-only model, raises `FeatureNotSupportedError`
before the request leaves your process. Use `client.supports()` to
pre-flight:

```python
client.supports("tools", model="deepseek-v4-flash")          # -> True
client.supports("vision", model="deepseek-v4-flash")         # -> False
client.supports("vision", provider="qwen", model="qwen-vl-max")  # -> True
```

## Result

A non-streaming call returns a `UnifiedResponse` with:

- `content` — the generated text
- `finish_reason` — standardized reason (`stop`, `length`, `tool_calls`, ...)
- `usage` — token usage (`UsageMetrics`)
- `tool_calls` — parsed tool calls, if any
- `parsed` — validated structured output (populated when the response is
  parsed with `output_schema`; see [structured_output.md](structured_output.md))
- `raw` — the original provider payload
