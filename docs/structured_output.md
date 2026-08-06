# Structured Output

> Stub — content to be written.

## Overview

The SDK can enforce structured JSON output from LLM responses by validating the output against a Pydantic model.

## Usage

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

result = client.chat(
    prompt="Summarize this article: ...",
    output_schema=Summary,
    model="deepseek-chat",
)

print(result.parsed.title)
print(result.parsed.key_points)
print(result.parsed.word_count)
```

## How It Works

1. The Pydantic schema is converted to a JSON Schema.
2. The JSON Schema is injected into the provider's tool-calling / structured output parameter.
3. The provider is instructed to output JSON conforming to the schema.
4. The SDK validates the returned JSON against the schema.
5. On validation failure: `ResponseParsingError` is raised (catchable by `RetryMiddleware`).