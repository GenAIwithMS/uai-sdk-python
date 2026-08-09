# Tools

The SDK supports function calling / tool use via a provider-agnostic tool definition format. Tool execution itself is the application's responsibility (per OpenAI guidance).

## Defining Tools

Tools use the OpenAI-compatible format — a dict (or
`uai.models.ToolDefinition`) per tool:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]

result = client.chat(
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
    model="deepseek-chat",
)
```

## Handling Tool Calls

When the model decides to call a tool, the response contains `tool_calls`.
The arguments are delivered as a **JSON string** — use `get_arguments()` to
parse them into a dict:

```python
if result.tool_calls:
    for call in result.tool_calls:
        # Execute the tool (arguments is a JSON string; parse it first)
        args = call.get_arguments()
        observation = get_weather(args["city"])
        # Feed the result back; a "tool" message must reference the
        # original tool call via tool_call_id
        result = client.chat(
            messages=[
                {"role": "user", "content": "What's the weather in Beijing?"},
                {"role": "assistant", "tool_calls": [call]},
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation,
                },
            ],
            model="deepseek-chat",
        )
```

## Streaming tool calls

Tool-call arguments can arrive across multiple streamed chunks
(`delta.tool_calls`); each `StreamChunk` exposes them on
`chunk.tool_calls`.

## MCP Integration

The SDK accepts OpenAI-style tool definitions directly. Model Context
Protocol (MCP) server integration is **not implemented yet** — expose MCP
tools by converting them to the OpenAI-compatible format above before
passing them to `client.chat()`.
