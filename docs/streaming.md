# Streaming

> Stub — content to be written.

## Overview

Streaming returns response chunks as Server-Sent Events (SSE) flow in from the provider, enabling real-time UI updates.

## Usage

```python
chunks = client.chat(
    prompt="Write a 5-paragraph essay.",
    model="qwen-plus",
    stream=True,
)

for chunk in chunks:
    print(chunk.content, end="", flush=True)

print()  # newline at end
print(f"Finish reason: {chunk.finish_reason}")
```

## Time-To-First-Token (TTFT)

Each chunk exposes timing metadata:

```python
for chunk in streaming_response:
    print(f"  TTFB chunk: {chunk.ttft_ms}ms")
```

## Aggregation

Set `aggregate=True` to collect the full response at the end:

```python
full = client.chat(
    prompt="Hello",
    stream=True,
    aggregate=True,
)
print(full.content)  # full assembled text
```