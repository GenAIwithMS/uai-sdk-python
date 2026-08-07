# Universal AI Provider SDK (uai-sdk-python)

A universal, modular AI infrastructure layer for Python that abstracts multiple Chinese LLM providers behind a single, stable API with an opt-in middleware architecture.

## Installation

> 🚧 **Under Active Development** — This SDK has **NOT** been released on PyPI yet. We're targeting our first release for **Q3 2026**.

### Local Install (Development Mode)

```bash
# Clone the repository
git clone https://github.com/uai-sdk/uai-sdk-python.git
cd uai-sdk-python

# Install in development mode
pip install -e .
```


Once published to PyPI, the command will be:
```bash
pip install uai-sdk
```

## Quick Start

```python
from uai import UniversalAI

client = UniversalAI(
    providers=["deepseek", "qwen"],
    api_keys={"deepseek": "sk-...", "qwen": "sk-..."},
)

# Chat
result = client.chat(
    prompt="Translate to English: 你好",
    model="deepseek-chat",
)
print(result.content)

# Streaming
for chunk in client.chat(prompt="Tell me a story...", stream=True):
    print(chunk.content, end="", flush=True)

# Structured output
result = client.chat(
    prompt="Extract the key points",
    output_schema=MyPydanticModel,
)
print(result.parsed)
```

## Features

- ✅ Unified API across multiple LLM providers
- ✅ Chat, streaming, tool-calling, structured outputs
- ✅ Embeddings (`client.embed`) and rerank (`client.rerank`) via provider adapters
- ✅ Vision via chat content blocks (image_url) on vision-capable models
- ✅ Modular middleware (retry, cache, logging, tracing)
- ✅ Provider adapters with strict capability enforcement
- ✅ OpenTelemetry metrics and tracing
- ✅ Security-first design (no secret logging, input validation)
- ✅ Extensible — easily add new providers via the [PDK](docs/pdk.md)
- ⏳ Audio / TTS / transcription — **not yet implemented** (deferred)

## Supported Providers

| Provider   | Chat | Streaming | Tools | Vision | Embeddings | Rerank | Audio |
|------------|------|-----------|-------|--------|------------|--------|-------|
| DeepSeek   | ✅   | ✅        | ✅    | ❌     | ✅         | ❌     | ❌    |
| Qwen       | ✅   | ✅        | ✅    | ✅     | ✅         | ✅     | ❌    |
| GLM        | ✅   | ✅        | ✅    | ❌     | ✅         | ✅     | ❌    |
| Kimi       | ✅   | ✅        | ✅    | ❌     | ❌         | ❌     | ❌    |
| StepFun    | ✅   | ✅        | ✅    | ✅     | ✅         | ❌     | ❌    |
| Doubao     | ✅   | ✅        | ✅    | ✅     | ✅         | ❌     | ❌    |
| MiniMax    | ✅   | ✅        | ✅    | ✅     | ✅         | ❌     | ❌    |
| Hunyuan    | ✅   | ✅        | ✅    | ✅     | ✅         | ❌     | ❌    |

---

## Documentation

- [Architecture](docs/architecture.md) — middleware pipeline, adapter contract, UnifiedRequest lifecycle
- [Chat](docs/chat.md) — conversational completions, history management, system prompts
- [Streaming](docs/streaming.md) — SSE handling, TTFT, chunk aggregation
- [Tools](docs/tools.md) — function calling & MCP integration
- [Embeddings](docs/embeddings.md) — text embedding operations via adapters
- [Vision](docs/vision.md) — multimodal image interpretation via chat content blocks
- [Rerank](docs/rerank.md) — document ranking (Qwen, GLM)
- [Structured Output](docs/structured_output.md) — schema validation & parsing
- [Middleware](docs/middleware.md) — creating and composing interceptors
- [Telemetry](docs/telemetry.md) — OpenTelemetry, Prometheus, metric conventions
- [Configuration](docs/configuration.md) — env vars, YAML config, API key management
- [PDK](docs/pdk.md) — Provider Development Kit guide

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.


### Issue Label Policy

All new issues are automatically evaluated and tagged:

| Label              | Meaning                                           |
|--------------------|---------------------------------------------------|
| `roadmap-aligned`  | On our planned roadmap — we intend to build it.   |
| `needs-discussion` | Requires design discussion before we decide.      |
| `won't-implement`  | Out of scope for this project's direction.        |
| `help-wanted`      | We explicitly welcome PRs for this.               |

See [`.github/LABELS.md`](.github/LABELS.md) for the full label set and conventions.

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
