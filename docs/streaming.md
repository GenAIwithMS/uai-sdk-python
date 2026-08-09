# Streaming

> Streaming returns response chunks as Server-Sent Events (SSE) flow in from the provider, enabling real-time UI updates.

## Usage

```python
chunks = client.chat(
    messages=[{"role": "user", "content": "Write a 5-paragraph essay."}],
    model="qwen-plus",
    stream=True,
)

for chunk in chunks:
    print(chunk.content, end="", flush=True)

print()  # newline at end
print(f"Finish reason: {chunk.finish_reason}")
```

With `stream=True`, `client.chat()` returns an **iterator of `StreamChunk`
objects**. Each chunk carries a delta of text content, and the final chunk
has `is_final=True` and a `finish_reason`. A `stream_callback` can be passed
instead of iterating:

```python
client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    stream_callback=lambda chunk: print(chunk.content, end="", flush=True),
)
```

## Time-To-First-Token (TTFT)

The **first chunk that carries content** also exposes timing metadata:

```python
for chunk in streaming_response:
    if chunk.ttft_ms is not None:
        print(f"TTFT: {chunk.ttft_ms}ms")
        break
```

`ttft_ms` (time-to-first-token, in milliseconds) is populated only on the
first content chunk; subsequent chunks report `None`.

## Aggregation

There is no built-in `aggregate` flag — assemble the full text by
accumulating chunk content yourself:

```python
full_text = ""
for chunk in client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    if chunk.content:
        full_text += chunk.content

print(full_text)
```
