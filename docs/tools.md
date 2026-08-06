# Tools

> Stub — content to be written.

## Overview

The SDK supports function calling / tool use via a provider-agnostic tool definition format. Tool execution itself is the application's responsibility (per OpenAI guidance).

## Defining Tools

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
    prompt="What's the weather in Beijing?",
    tools=tools,
    model="deepseek-chat",
)
```

## Handling Tool Calls

```python
if result.tool_calls:
    for call in result.tool_calls:
        # Execute the tool
        observation = get_weather(call.arguments["city"])
        # Feed result back
        result = client.chat(
            messages=[
                {"role": "user", "content": "What's the weather in Beijing?"},
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "name": "get_weather", "content": observation},
            ],
            model="deepseek-chat",
        )
```

## MCP Integration

See [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk) for defining tools as MCP servers. The SDK accepts MCP-formatted tool definitions out of the box.