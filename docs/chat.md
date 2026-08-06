# Chat

> Stub — content to be written.

## Basic Usage

```python
result = client.chat(
    prompt="Explain quantum computing in 2 sentences.",
    model="deepseek-chat",
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

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | Simple prompt string (convenience) |
| `messages` | `list[dict]` | Full conversational history |
| `model` | `str` | Model name (must match provider) |
| `max_tokens` | `int` | Max output tokens |
| `temperature` | `float` | Sampling temperature |
| `tools` | `list[dict]` | Function/tool definitions |
| `output_schema` | `pydantic.BaseModel` | For structured output |
| `stream` | `bool` | Enable streaming |