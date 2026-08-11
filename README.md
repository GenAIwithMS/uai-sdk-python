<div align="center">

# Universal AI Provider SDK

**One consistent Python interface across eight Chinese LLM providers — with an opt-in middleware pipeline, strict capability enforcement, and zero vendor lock-in.**

[![PyPI version](https://img.shields.io/pypi/v/uai-sdk?color=2f7bff&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/uai-sdk/)
[![Python versions](https://img.shields.io/pypi/pyversions/uai-sdk?color=2f7bff&logo=python&logoColor=white)](https://pypi.org/project/uai-sdk/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f7bff.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/GenAIwithMS/uai-sdk-python/ci.yml?branch=main&label=ci&logo=githubactions&logoColor=white)](https://github.com/GenAIwithMS/uai-sdk-python/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/types-mypy-1f5082.svg)](https://mypy-lang.org/)

[Installation](#-installation) · [Quick Start](#-quick-start) · [Providers](#-supported-providers) · [Middleware](#-middleware) · [CLI](#-cli) · [Docs](#-documentation)

</div>

---

## Why UAI

Every Chinese LLM provider ships its own SDK, its own request shape, its own error taxonomy, and its own idea of what "streaming" means. Swapping providers means rewriting your integration layer.

`uai-sdk` gives all of them one stable surface. You build **one client per provider**, and every client behaves identically:

```python
deepseek = UniversalAI(provider="deepseek")
qwen     = UniversalAI(provider="qwen")

deepseek.chat(messages=[...])    # same call signature
qwen.chat(messages=[...])        # same UnifiedResponse back
```

Same `UnifiedRequest` in, same `UnifiedResponse` out. Same exception hierarchy. Same middleware. The adapter layer absorbs the differences, so switching providers is a configuration change rather than a rewrite.

| | |
|---|---|
| 🔌 **8 providers, 1 interface** | DeepSeek, Qwen, GLM, Kimi, StepFun, Doubao, MiniMax, Hunyuan |
| 🛡️ **Fail fast, not late** | The capability matrix rejects unsupported features *before* the network call — no wasted round-trips, no cryptic 400s |
| 🧩 **Opt-in middleware** | Retry, cache, circuit breaker, logging, metrics, tracing — composed with `client.use(...)`, zero cost when unused |
| 📐 **Typed end to end** | Pydantic v2 models throughout; `mypy`-clean public surface |
| ⚡ **Measured, not claimed** | < 5 ms SDK overhead and ≥ 1,000 req/min enforced by a KPI regression suite in CI |
| 🪶 **Four dependencies** | `pydantic`, `httpx`, `pyyaml` — lazily-imported adapters keep cold start small |
| 🧪 **Offline-testable** | A stdlib-only `MockProviderServer` exercises the full pipeline without credentials |
| 🔧 **Extensible** | Add a provider without forking the SDK — see the [PDK](docs/pdk.md) |

---

## 📦 Installation

```bash
pip install uai-sdk
```

<details>
<summary>Other install methods</summary>

```bash
# Poetry
poetry add uai-sdk

# uv
uv add uai-sdk

# From source (development)
git clone https://github.com/GenAIwithMS/uai-sdk-python.git
cd uai-sdk-python
poetry install --with dev
```

</details>

**Requirements:** Python 3.9 – 3.13 · `pydantic ^2.5` · `httpx ^0.27` · `pyyaml ^6.0`

---

## 🚀 Quick Start

Set the API key for the provider you want, then go:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

```python
from uai import UniversalAI

client = UniversalAI(provider="deepseek", model="deepseek-v4-flash")

response = client.chat(
    messages=[{"role": "user", "content": "Translate to English: 你好"}],
)

print(response.content)          # "Hello"
print(response.usage.total_tokens)
print(response.finish_reason)    # FinishReason.STOP
```

The API key is read from the provider's environment variable when `api_key` is omitted. Pass it explicitly if you'd rather manage credentials yourself:

```python
client = UniversalAI(api_key="sk-...", provider="deepseek")
```

> **Credentials are scoped to one provider.** An `api_key` you pass to the constructor is used for that client's provider and no other — it is never reused as a fallback for a different one. Build a separate client per provider, each with its own key.

### Choosing a model

`model=` takes any model id the provider accepts, the same way per-provider
LangChain classes do:

```python
client = UniversalAI(provider="deepseek", model="deepseek-v4-pro")

client.chat(messages=[...], model="deepseek-v4-flash")   # per-call override

client = UniversalAI(model="glm-4.7")                    # provider inferred → "glm"
```

**A model the registry doesn't know is still sent.** Provider catalogues move
faster than this package is released, so an unrecognised id is forwarded
verbatim (with a warning) rather than rejected:

```python
# Works on day one of a new release, no SDK upgrade needed.
client = UniversalAI(provider="deepseek", model="deepseek-v4-flash-0731")
```

Two things are still refused, because both are mistakes rather than new
releases: a model belonging to a **different** registered provider, and any
unknown id when you pass `strict_models=True`. To give a new model real
metadata instead of just passing it through, declare it in a `providers.yaml` —
see [docs/configuration.md](docs/configuration.md#registering-a-new-model).

> **`.env` files are not read by the SDK.** It reads the process environment.
> Load the file yourself first — `pip install "uai-sdk[dotenv]"`, then
> `load_dotenv()` before constructing a client. To set a model from the
> environment, use `UAI_PROVIDER_DEEPSEEK_DEFAULT_MODEL` (there is no bare
> `DEEPSEEK_MODEL` variable).

### Streaming

```python
for chunk in client.chat(messages=[{"role": "user", "content": "Tell me a story"}], stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.ttft_ms:
        print(f"\n[time to first token: {chunk.ttft_ms:.0f} ms]")
```

### Structured output

Hand it a Pydantic model; get a validated instance back on `.parsed`.

```python
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    tags: list[str]
    summary: str

response = client.chat(
    messages=[{"role": "user", "content": "Summarize the article: ..."}],
    output_schema=Article,
)

article = response.parsed          # -> Article
print(article.tags)
```

Malformed JSON raises `ResponseParsingError` rather than silently handing you a broken object. Streaming works too — the terminal chunk carries `.parsed`.

### Tool calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat(
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
)

for call in response.tool_calls or []:
    args = call.get_arguments()        # JSON string -> dict
    observation = get_weather(args["city"])
```

### Vision

```python
from uai.models import ChatMessage, ImageContent, ImageURL, Role

client = UniversalAI(provider="qwen", model="qwen-vl-max")

response = client.chat(messages=[
    ChatMessage(role=Role.USER, content=[
        ImageContent(image_url=ImageURL(url="https://example.com/photo.jpg")),
    ]),
    ChatMessage(role=Role.USER, content="Describe this image."),
])
```

Send an image to a text-only model and you get `FeatureNotSupportedError` **before** the request leaves the process.

### Embeddings & rerank

```python
qwen = UniversalAI(provider="qwen")     # reads DASHSCOPE_API_KEY

response = qwen.embed(["hello world", "你好世界"], model="text-embedding-v4")
print(response.vectors[0].dimension)
print(response.vectors[0].values[:5])

ranked = qwen.rerank(
    query="What is machine learning?",
    documents=["ML is a field of AI.", "Paris is in France."],
    model="qwen3-rerank",
)
print(ranked.results[0].index)     # most relevant document
```

### Pre-flight capability checks

```python
client.supports("vision")                              # active provider/model
client.supports("rerank", provider="glm")              # ask about another one
```

---

## 🌐 Supported Providers

| Provider | Default model | API key env var | Chat | Stream | Tools | Vision | Embed | Rerank | Reasoning |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DeepSeek** | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | ✅ | ✅ | ✅ | — | — | — | ✅ |
| **Qwen** (Model Studio) | `qwen3.7-plus` | `DASHSCOPE_API_KEY` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| **GLM** (Zhipu) | `glm-4.7` | `BIGMODEL_API_KEY` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MiniMax** | `MiniMax-M3` | `MINIMAX_API_KEY` | ✅ | ✅ | ✅ | — | ✅ | — | — |
| **Kimi** (Moonshot) | `kimi-k3` | `MOONSHOT_API_KEY` | ✅ | ✅ | ✅ | ✅ | — | — | ✅ |
| **StepFun** | `step-3.7-flash` | `STEPFUN_API_KEY` | ✅ | ✅ | ✅ | ✅ | — | — | ✅ |
| **Doubao** (Volcengine Ark) | `doubao-seed-2-0-pro` | `ARK_API_KEY` | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **Hunyuan** (Tencent) | `hunyuan-turbo-latest` | `HUNYUAN_API_KEY` | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |

Capabilities are declared **per model**, not per provider — the table shows the union across each provider's model family. Run `uai list-models <provider>` for the exact per-model matrix, or see [docs/providers.md](docs/providers.md).

> **Audio, TTS, and transcription are not implemented.** Every provider reports these as `False`, and requesting them raises `FeatureNotSupportedError`. They are deliberately deferred — see the [roadmap](#-roadmap).

---

## 🧩 Middleware

Middleware is **opt-in**. A client with no middleware registered runs no pipeline code at all.

```python
from uai import UniversalAI
from uai.middleware import (
    CacheMiddleware,
    CircuitBreakerMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    RetryMiddleware,
)

client = (
    UniversalAI(provider="deepseek")
    .use(CircuitBreakerMiddleware(failure_threshold=5, reset_timeout=30.0))
    .use(RetryMiddleware(max_retries=3, base_delay=0.5))
    .use(CacheMiddleware(ttl=300))
    .use(LoggingMiddleware())
)
```

`before_request` hooks run in registration order; `after_response` and `on_error` run in reverse — so the outermost middleware sees the request first and the response last.

| Middleware | What it does |
|:---|:---|
| `RetryMiddleware` | Exponential backoff with jitter on 429/5xx, network errors, and timeouts. Never retries auth failures. Streaming retries only before the first chunk reaches you. |
| `CacheMiddleware` | In-memory TTL cache keyed on a normalized request hash. Schema-aware, so two requests differing only by `output_schema` never collide. Streaming bypasses it. |
| `CircuitBreakerMiddleware` | Per `(provider, model)` closed → open → half-open state machine. Fast-fails a degraded provider instead of burning retries against it. |
| `LoggingMiddleware` | Structured request/response logging. Never logs API keys or credentials. |
| `MetricsMiddleware` | Prometheus-style counters, gauges, and histograms via an in-process `MetricsRegistry` with `.render()`. |
| `TracingMiddleware` | OpenTelemetry GenAI semantic-convention spans, recorded through a dependency-free `SpanRecorder`. |

Writing your own is one subclass:

```python
from uai.middleware import BaseMiddleware

class AuditLogger(BaseMiddleware):
    name = "audit"

    def before_request(self, request, context):
        audit.record(context.request_id, context.provider, context.model)
        return request

    def after_response(self, response, context):
        audit.complete(context.request_id, elapsed_ms=context.elapsed_ms)
        return response

client.use(AuditLogger())
```

See [docs/middleware.md](docs/middleware.md) for the full hook contract.

---

## ⚙️ Configuration

Three layers, highest precedence first: **constructor arguments → environment variables → YAML/JSON config file → built-in registry defaults.**

```python
client = UniversalAI(
    provider="qwen",
    model="qwen3.7-plus",
    api_key="sk-...",
    timeout=60.0,       # seconds; overrides the registry and env config
    max_retries=3,      # enables retries — see below
)
```

`max_retries` is shorthand for registering a `RetryMiddleware`. It is composed **inside** everything you add with `use()`, so a circuit breaker still short-circuits without consuming attempts and a cache hit skips retrying entirely. Leave it unset for no retries — retrying stays opt-in. If you register a `RetryMiddleware` explicitly, that wins and the shorthand is dropped (nesting both would multiply your request count).

<details>
<summary>Environment variable overrides</summary>

Any registry field can be overridden without touching code:

| Variable | Purpose |
|:---|:---|
| `UAI_PROVIDER_{NAME}_BASE_URL` | Point a provider at a proxy, gateway, or mock server |
| `UAI_PROVIDER_{NAME}_TIMEOUT` | Request timeout in seconds |
| `UAI_PROVIDER_{NAME}_MAX_RETRIES` | Retry ceiling |
| `UAI_PROVIDER_{NAME}_RATE_LIMIT_RPM` | Requests-per-minute limit |
| `UAI_PROVIDER_{NAME}_RATE_LIMIT_TPM` | Tokens-per-minute limit |
| `UAI_PROVIDER_{NAME}_API_KEY_ENV` | Rename the variable holding the key |
| `UAI_PROVIDER_{NAME}_DISABLE_{CAPABILITY}` | Force-disable a capability (e.g. kill-switch a broken vision endpoint) |
| `UAI_CONFIG_PATH` | Path to a YAML/JSON config file |

```bash
export UAI_PROVIDER_DEEPSEEK_BASE_URL="https://my-gateway.internal/v1"
export UAI_PROVIDER_QWEN_TIMEOUT="60"
export UAI_PROVIDER_GLM_DISABLE_RERANK="1"
```

</details>

<details>
<summary>Config file</summary>

```yaml
# uai.yaml  — discovered via UAI_CONFIG_PATH or the default search paths
providers:
  deepseek:
    base_url: https://my-gateway.internal/v1
    timeout: 60.0
    max_retries: 5
  my-internal-llm:
    display_name: Internal Model Gateway
    base_url: https://llm.internal/v1
    api_key_env_var: INTERNAL_LLM_KEY
    default_model: internal-chat
```

User-supplied providers are deep-merged with the built-in registry — you override what you need and inherit the rest.

</details>

Full details in [docs/configuration.md](docs/configuration.md).

---

## 🖥️ CLI

Installing the package puts a `uai` command on your PATH.

```bash
uai list-providers                      # registered providers + whether a key is configured
uai list-models deepseek                # models, context windows, capabilities
uai benchmark                           # benchmark every chat model you have a key for
uai benchmark --providers deepseek,qwen --iterations 5 --json
```

```
Provider   Model                  Iter   OK  TTFT ms   Lat ms   Tok/s    Cost $  Err %
------------------------------------------------------------------------------------
deepseek   deepseek-v4-flash             5    5      312      894    41.2  0.000418   0.0%
qwen       qwen3.7-plus                 5    5      408     1102    33.7  0.000356   0.0%
```

TTFT is measured from real streamed responses; cost is derived from the registry's per-model pricing and actual token usage. See [docs/benchmark.md](docs/benchmark.md).

---

## 🧪 Testing Without Credentials

`uai.testing.MockProviderServer` is a stdlib-only, in-process HTTP server that speaks the OpenAI-compatible wire protocol — so your tests exercise the real client → middleware → adapter → HTTP → parse path, offline.

```python
import os
from uai import UniversalAI
from uai.testing import MockProviderServer

with MockProviderServer(latency_ms=5.0) as server:
    os.environ["UAI_PROVIDER_DEEPSEEK_BASE_URL"] = server.base_url
    client = UniversalAI(api_key="test", provider="deepseek")

    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    assert response.content

    server.fail_with(429, count=2)      # inject transient failures, then recover
```

---

## 📊 Performance

Non-functional guarantees are enforced by a KPI regression suite that runs in CI on every push.

| KPI | Target |
|:---|:---|
| SDK processing overhead per request | **< 5 ms** |
| Streaming TTFT handling overhead | **< 30 ms** |
| Throughput | **≥ 1,000 req/min** |
| Idle memory footprint (marginal over interpreter) | **< 50 MB** |
| Memory under ~100 parallel requests | **< 150 MB** |

Provider adapters are imported lazily, so an application using one provider never pays the import cost of the other seven. Details and tuning knobs in [docs/performance.md](docs/performance.md).

---

## 🚨 Error Handling

Every SDK failure derives from `UAIError`, so one `except` clause is enough — and each subclass carries `provider`, `model`, `status_code`, and `response_body` for triage.

```python
from uai import UAIError, UAIRateLimitError, FeatureNotSupportedError

try:
    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
except FeatureNotSupportedError as e:
    print(f"{e.feature} unavailable on {e.provider}/{e.model}")
except UAIRateLimitError as e:
    time.sleep(e.retry_after or 5)
except UAIError as e:
    print(f"[{e.status_code}] {e}")
```

```
UAIError
├── UAIAuthenticationError      401 / invalid credentials
├── UAIRateLimitError           429 — carries .retry_after
├── UAINetworkError             connection failures
├── UAITimeoutError             request exceeded its timeout
├── UAICircuitOpenError         circuit breaker rejected the call
├── ResponseParsingError        malformed or schema-invalid response
├── FeatureNotSupportedError    capability matrix rejection
├── ConfigError                 invalid provider configuration
└── UAIErrorGroup               aggregated failures across providers
```

---

## 📚 Documentation

<table>
<tr><td valign="top" width="50%">

**Getting started**
- [Architecture](docs/architecture.md) — pipeline, adapter contract, request lifecycle
- [Configuration](docs/configuration.md) — keys, env vars, YAML config
- [Providers](docs/providers.md) — capability matrix & provider notes

**Core API**
- [Chat](docs/chat.md)
- [Streaming](docs/streaming.md)
- [Structured Output](docs/structured_output.md)
- [Tools](docs/tools.md)

</td><td valign="top" width="50%">

**More capabilities**
- [Vision](docs/vision.md)
- [Embeddings](docs/embeddings.md)
- [Rerank](docs/rerank.md)

**Operating it**
- [Middleware](docs/middleware.md)
- [Telemetry](docs/telemetry.md)
- [Benchmark CLI](docs/benchmark.md)
- [Performance](docs/performance.md)
- [PDK](docs/pdk.md) — add your own provider

</td></tr>
</table>

---

## 🗺️ Roadmap

| Status | Item |
|:---:|:---|
| ✅ | Unified client, registry, and capability enforcement |
| ✅ | Eight provider adapters — chat, streaming, tools, vision, embeddings, rerank |
| ✅ | Middleware engine — retry, cache, circuit breaker, logging, metrics, tracing |
| ✅ | Structured output with Pydantic validation |
| ✅ | Benchmark CLI and KPI regression suite |
| 🚧 | Async client (`AsyncUniversalAI`) |
| 🚧 | Automatic cross-provider fallback and routing |
| ⏳ | Audio, TTS, and transcription |
| ⏳ | Persistent cache backends (Redis) |

Have an opinion on the order? [Open an issue](https://github.com/GenAIwithMS/uai-sdk-python/issues/new/choose) — every issue is triaged against the roadmap.

---

## 🤝 Contributing

Contributions are welcome. Start with the [Contributing Guide](CONTRIBUTING.md), and read the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

Adding a provider is the highest-leverage contribution and doesn't require touching the core — the [Provider Development Kit](docs/pdk.md) walks through the adapter contract end to end.

Issues are labeled so you always know where a request stands:

| Label | Meaning |
|:---|:---|
| `roadmap-aligned` | On the plan — we intend to build it |
| `needs-discussion` | Requires design discussion first |
| `won't-implement` | Out of scope for this project's direction |
| `help-wanted` | We explicitly welcome a PR |
| `good-first-issue` | Beginner-friendly |

Full conventions in [`.github/LABELS.md`](.github/LABELS.md).

---

## 📝 Changelog

Release history lives in [CHANGELOG.md](CHANGELOG.md), following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## 📄 License

Released under the [MIT License](LICENSE). © 2026 Muhammad Siddiq.

<div align="center">
<sub>Built for developers who'd rather ship than rewrite their provider integration.</sub>
</div>
