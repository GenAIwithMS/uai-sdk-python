# Changelog

All notable changes to the **Universal AI Provider SDK** (`uai-sdk`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Maintainers: add entries under [Unreleased] as changes land, then promote the
section to a new version heading on release. Categories, in order:
Added · Changed · Deprecated · Removed · Fixed · Security
-->

## [Unreleased]

### Added

- Nothing yet.

---

## [0.2.0] — 2026-08-11

A model-resolution release. The registry stops being an allowlist that could
block valid models, the model catalogue is refreshed against vendor
documentation, and several silent-drop bugs in request construction are fixed.

**Migration in one line:** if you pinned `deepseek-chat` or `deepseek-reasoner`,
nothing breaks — they now resolve to `deepseek-v4-flash` — but you should move
to the V4 ids, because the vendor retired the old ones on 2026-07-24.

### Fixed

- **The client-wide default model leaked across providers.** `UniversalAI` stored a single `_default_model` string taken from the constructor provider, then used it as the fallback for *every* provider the client called. A client built for DeepSeek that issued `client.chat(..., provider="qwen")` resolved the DeepSeek model against Qwen's catalogue and raised `ValueError: Model 'deepseek-chat' not found for provider 'qwen'`. The same fault hit `embed()`, `rerank()`, `supports()` and streaming chat — every public entry point that accepts a `provider=` override.

  Defaults are now keyed by provider. A constructor `model=` belongs to the provider it was supplied with; any other provider falls back to its own default.

- **`embed()` and `rerank()` inherited the *chat* default model.** `UniversalAI(provider="qwen").embed("hi")` raised `FeatureNotSupportedError` for `qwen-plus` even though Qwen ships `text-embedding-v4` and `qwen3-rerank`. `ProviderConfig` gained `default_embedding_model` and `default_rerank_model`, and resolution falls back to the first model advertising the required capability.

- **A constructor `model=` was never validated.** `UniversalAI(provider="deepseek", model="gpt-4o")` constructed successfully and failed only at the first `chat()`. Worse, `model="qwen-plus"` with `provider="deepseek"` was silently accepted. Both are now caught at construction, and a model belonging to a different registered provider produces an error naming that provider.

- **Generation parameters were silently dropped.** `frequency_penalty`, `presence_penalty` and `user` were valid `UnifiedRequest` fields that `_build_request_body` never serialized, so callers set them and the provider never saw them.

- **Unknown `chat()` keyword arguments were silently discarded.** `client.chat(..., seed=42)` dropped `seed` and sent the request anyway. Unrecognised names now raise `TypeError` listing the supported fields.

- **The chat path bypassed the provider adapters entirely.** `_build_request_body` hand-rolled an OpenAI-shaped body and posted to a hardcoded `/chat/completions`, so every adapter's `format_request` was dead code — MiniMax's content-block flattening and the penalty/user fields among them. Chat now routes through `adapter.format_request` and an overridable `chat_path`.

- **The config-file loader was never invoked.** `loader.py` documented that `apply_to_registry` ran "automatically during `UniversalAI` initialisation"; it did not, so a `providers.yaml` on disk had no effect. The client now discovers and applies the file at construction, with a new `config_path=` argument for an explicit path.

- **Environment overrides bypassed schema validation.** `apply_env_overrides_to_config` used `model_copy(update=...)`, which does not re-run validators, so an invalid override surfaced later inside a request. The config is now reconstructed and re-validated, and an invalid override is warned about and ignored.

- **Model-id resolution was implemented three times** — in the client, the enforcer and the schema — with divergent error messages, and `client._resolve_model(provider, model_id)` ignored its `model_id` argument entirely. All resolution now funnels through `ProviderConfig.resolve_model`.

- **Adapters carried their own `_DEFAULT_MODEL` constants**, a third source of truth that had drifted (DeepSeek's still named the retired `deepseek-chat`). Adapters now read the registry via `BaseProviderAdapter.default_model()`.

- Middleware labels (metrics, cache keys, traces) now use the **canonical** model id, so an alias and its target aggregate as one series instead of splitting into two.

### Added

- **Unknown model ids are forwarded to the provider** instead of raising, so a model released after this SDK version is usable immediately — `UniversalAI(model="deepseek-v4-flash-0731")` works with no upgrade. Capability checks for such an id fall back to the provider's aggregate capabilities, and a warning is logged. Opt out with `strict_models=True` or `UAI_PROVIDER_{NAME}_ALLOW_UNKNOWN_MODELS=false`.
- **Provider inference from the model id.** `UniversalAI(model="glm-4.7")` resolves to the `glm` provider when the id maps to exactly one registered provider; ambiguity and unknown ids raise rather than guess.
- **`UAI_PROVIDER_{NAME}_DEFAULT_MODEL`**, `_DEFAULT_EMBEDDING_MODEL`, `_DEFAULT_RERANK_MODEL` and `_ALLOW_UNKNOWN_MODELS` environment overrides. Previously no environment variable could set the model at all.
- **`ModelNotFoundError`** (a `UAIError`), replacing bare `ValueError`s that could not be caught through the SDK's own exception hierarchy. It distinguishes an unknown id from one belonging to another provider, and names the remedy.
- `ProviderConfig.resolve_model`, `knows_model`, `default_model_for`, and `find_providers_for_model` in the registry API.
- An optional `dotenv` extra (`pip install "uai-sdk[dotenv]"`) plus explicit documentation that **the SDK does not read `.env` files** — it reads the process environment, so a `.env` must be loaded by the application first.

### Changed

- **The model catalogue was rebuilt against vendor documentation (2026-08).** The previous registry contained model ids that do not exist and capability claims with no backing — `deepseek-chat` advertising `embeddings`, StepFun's *vision chat model* advertising `embeddings`, a `glm-4.7` alias pointing at the unrelated `glm-4.5v`.

  | Provider | Was (default) | Now (default) |
  |---|---|---|
  | DeepSeek | `deepseek-chat` | `deepseek-v4-flash` |
  | Qwen | `qwen-plus` | `qwen3.7-plus` |
  | GLM | `glm-4.7` | `glm-4.7` |
  | Kimi | `kimi-k2.5` | `kimi-k3` |
  | StepFun | `stepfun-2.5` | `step-3.7-flash` |
  | Doubao | `doubao-pro-32k` | `doubao-seed-2-0-pro` |
  | MiniMax | `minimax-m2.5` | `MiniMax-M3` |
  | Hunyuan | `hunyuan-turbo` | `hunyuan-turbo-latest` |

  DeepSeek, Qwen, Kimi, MiniMax and GLM entries are verified against vendor docs; StepFun, Doubao and Hunyuan are best-effort and marked as such in `providers.py`. Kimi's base URL moved to `api.moonshot.ai`, MiniMax's to `api.minimax.io`, and Doubao's key variable is now `ARK_API_KEY`.

- **`pricing` is `0.0` wherever a per-token rate could not be verified.** Zero means *unknown*, not free; `uai.benchmark` cost figures are only meaningful for models with populated pricing. Fabricated prices were removed rather than carried forward.

- `provider=` on the constructor now defaults to `None` (inferred, or `deepseek`) instead of the literal `"deepseek"`. Existing positional and keyword usage is unaffected.

### Deprecated

- `deepseek-chat`, `deepseek-chat-latest`, `deepseek-reasoner` and `deepseek-reasoner-latest` are retained as **aliases** of `deepseek-v4-flash`, the successor the vendor named when the ids were discontinued on 2026-07-24. Thinking mode moved from a model id to a request parameter under V4, so the adapter no longer infers `reasoning_format` from the model name.

---

## [0.1.5] — 2026-08-10

A correctness, documentation, and packaging-metadata release.

### Security

- **Fixed a credential leak across providers.** `UniversalAI._get_api_key()` returned the credential supplied to the constructor for *any* provider, ignoring which provider was actually being called. A client built as `UniversalAI(api_key="sk-deepseek", provider="deepseek")` that then issued `client.chat(..., provider="qwen")` transmitted the DeepSeek key to DashScope's API in the `Authorization` header — and likewise for `embed()` and `rerank()`, which accept the same `provider=` override.

  Constructor credentials are now **scoped to the client's default provider**. Any other provider resolves its own key from its `api_key_env_var`, and raises `ValueError` naming that variable when no key is available rather than falling back to an unrelated credential. Covered by regression tests in `tests/unit/test_client_credentials.py`.

### Fixed — client configuration

- **`UniversalAI(timeout=...)` had no effect.** The value was written to an internal `ProviderConfig` copy that no request path reads, so every request used the provider's registry default (30–45 s) regardless of what was passed. The timeout is now resolved at each of the four request sites — chat, streaming chat, `embed()`, and `rerank()` — and takes precedence over both the registry value and `UAI_PROVIDER_{NAME}_TIMEOUT`, matching the documented configuration precedence. It is held on the client rather than written into a config object, because `_resolve_model()` returns shared registry entries that other clients in the same process also read.
- **`UniversalAI(max_retries=...)` had no effect.** Nothing in the request path consumed it: retries live in `RetryMiddleware`, which takes its own independent count. The parameter now enables retries, as shorthand for registering a `RetryMiddleware`.

  The retry is composed **innermost**, beneath every middleware added through `use()`, which is the topology the middleware are documented to expect — an open circuit breaker short-circuits without consuming attempts, and a cache hit skips retrying entirely. Retrying remains **opt-in**: the `max_retries` value on each provider's registry entry does not switch it on, so behavior is unchanged for anyone not passing the parameter. An explicitly registered `RetryMiddleware` supersedes the shorthand and logs a warning, since composing both would nest two retry loops and multiply the request count.

### Changed

- API keys are now resolved at call time instead of being captured during construction, so a rotated environment variable takes effect without rebuilding the client.
- A `credentials` dict passed to the constructor is copied, so mutating the caller's dict afterwards no longer alters the client's credentials.

### Fixed

- **Corrected the declared license.** Package metadata declared `Apache-2.0` while the bundled `LICENSE` file is the MIT License. The metadata now correctly declares **MIT**, matching the actual license terms. The `License :: OSI Approved` classifier was updated accordingly.
- **Corrected the repository and documentation URLs.** Package metadata pointed at `github.com/uai-sdk/uai-sdk-python`, which does not exist. All links now resolve to the real repository at `github.com/GenAIwithMS/uai-sdk-python`.
- Removed the "not yet released on PyPI, targeting Q3 2026" notice from the README — the package has been published to PyPI since `0.1.0`.
- Fixed the roadmap anchor link in `docs/index.md`, which no longer resolved after the README was restructured.
- Corrected documentation that presented per-call `provider=` switching as the primary usage pattern. The SDK's model is one client per provider; examples in `README.md` and `docs/vision.md` now reflect that, and `docs/configuration.md` documents credential scoping explicitly.

### Added

- **`CHANGELOG.md`** — this file, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), with the `0.1.0` release history reconstructed from the commit log.
- **`CODE_OF_CONDUCT.md`** — the Contributor Covenant v2.1, which `CONTRIBUTING.md` previously linked to as an external document.
- `homepage` metadata, plus `Changelog` and `Issue Tracker` project URLs, so they appear in the PyPI sidebar.
- Links to the changelog and code of conduct from the documentation index.

### Changed

- **Rewrote `README.md`** for the released package: installation via `pip install uai-sdk`, runnable examples for chat, streaming, structured output, tool calling, vision, embeddings, and rerank; a provider matrix listing each provider's default model and API key environment variable; middleware, configuration, CLI, offline-testing, performance, and error-handling sections; and a roadmap.
- **Rewrote `CONTRIBUTING.md`** to describe the project as it actually exists. The previous version documented a source layout that had drifted from reality — it referenced `adapters/base.py` (now `adapters/base_adapter.py`), a `tests/integration/` directory and shared fixtures that were never added, and a `workflows/test.yml` that is now `ci.yml` and `publish.yml`. It also described changelog generation from commit messages, which has been replaced by this hand-maintained file. Added a step-by-step provider-contribution checklist and documented the real OIDC Trusted Publishing release process.
- Bumped the `Development Status` classifier from `2 - Pre-Alpha` to `3 - Alpha` to reflect the published release.

---

## [0.1.0] — 2026-08-09

Initial public release, published to [PyPI](https://pypi.org/project/uai-sdk/).

### Added

**Core client**

- `UniversalAI` client orchestrator — the single entry point for all provider interactions, with per-call `provider` and `model` overrides.
- Unified Pydantic v2 data models: `UnifiedRequest`, `UnifiedResponse`, `StreamChunk`, `ChatMessage`, `UsageMetrics`, `ToolDefinition`, `ToolCall`, `EmbeddingsResponse`, `RerankResponse`.
- `client.chat()` — chat completions with streaming (SSE) and non-streaming modes, an optional per-chunk `stream_callback`, and time-to-first-token measurement on `StreamChunk.ttft_ms`.
- `client.embed()` — text embeddings routed through provider adapters.
- `client.rerank()` — document relevance ranking routed through provider adapters.
- `client.supports()` — pre-flight capability checks before building a request.
- Vision support through multimodal chat content blocks (`ImageContent` / `ImageURL`).
- Tool calling with OpenAI-compatible tool definitions, including tool-call deltas across streamed chunks.
- Structured output — pass a Pydantic model as `output_schema` and receive a validated instance on `.parsed`, for both streaming and non-streaming responses.

**Providers**

- Eight provider adapters, all conforming to the `BaseProviderAdapter` contract: **DeepSeek**, **Qwen** (DashScope), **GLM** (Zhipu), **Kimi** (Moonshot), **StepFun**, **Doubao** (ByteDance), **MiniMax**, and **Hunyuan** (Tencent Cloud).
- Static provider registry with per-model metadata: context windows, max output tokens, pricing, aliases, and a per-model capability matrix.
- `CapabilityMatrixEnforcer` — merges registry capabilities with the adapter's own matrix and rejects unsupported features *before* any middleware or network work, raising `FeatureNotSupportedError`.
- Runtime provider registration via `register_provider()` for custom, self-hosted, or gateway-fronted endpoints.

**Middleware**

- `MiddlewareEngine` implementing the interceptor pattern: `before_request` in registration order, `after_response` / `on_error` in reverse, with `MiddlewareHalt` for short-circuiting.
- `RetryMiddleware` — exponential backoff with jitter for rate limits, network errors, timeouts, and 5xx responses; never retries authentication failures, and retries streaming calls only before the first chunk is delivered.
- `CacheMiddleware` — in-memory TTL cache keyed on a normalized request hash, with an `output_schema` fingerprint folded into the key so structured-output requests cannot collide.
- `CircuitBreakerMiddleware` — per `(provider, model)` closed/open/half-open state machine that fast-fails degraded providers.
- `LoggingMiddleware` — structured lifecycle logging that never emits credentials.
- `MetricsMiddleware` and `MetricsRegistry` — dependency-free, Prometheus-style counters, gauges, and histograms with a `render()` exposition format.
- `TracingMiddleware` and `SpanRecorder` — OpenTelemetry GenAI semantic-convention spans without a hard OpenTelemetry dependency.

**Configuration**

- Layered configuration resolution: constructor arguments → environment variables → YAML/JSON config file → registry defaults.
- Per-provider environment overrides (`UAI_PROVIDER_{NAME}_BASE_URL`, `_TIMEOUT`, `_MAX_RETRIES`, `_RATE_LIMIT_RPM`, `_RATE_LIMIT_TPM`, `_AUTH_TYPE`, `_API_KEY_ENV`, `_DISABLE_{CAPABILITY}`).
- Config-file loading with deep merge against built-in defaults, discovered via `UAI_CONFIG_PATH` or the default search paths.

**Tooling**

- `uai` command-line interface with `benchmark`, `list-providers`, and `list-models` subcommands.
- `uai benchmark` — offline-capable benchmarking across every configured chat model, reporting TTFT, latency, throughput, error rate, and estimated cost, in table or JSON form.
- `uai.testing.MockProviderServer` — a stdlib-only, in-process, OpenAI-compatible HTTP server supporting JSON and SSE responses, configurable latency, and injected transient failures.

**Quality**

- Full exception hierarchy rooted at `UAIError`, carrying `provider`, `model`, `status_code`, and `response_body`: `UAIAuthenticationError`, `UAIRateLimitError` (with `retry_after`), `UAINetworkError`, `UAITimeoutError`, `UAICircuitOpenError`, `ResponseParsingError`, `FeatureNotSupportedError`, `ConfigError`, `UAIErrorGroup`.
- KPI regression suite enforcing < 5 ms SDK overhead, < 30 ms streaming TTFT overhead, ≥ 1,000 req/min throughput, and bounded idle and sustained memory footprints.
- CI across Python 3.9–3.12: Ruff lint and format checks, MyPy type checking, the full test suite with coverage, plus `bandit` and `pip-audit` security scanning.

### Performance

- Provider adapters are imported lazily on first use rather than at `import uai`, so applications using a single provider avoid roughly 7 MB of marginal memory and the import cost of the other seven adapter modules.

### Security

- API keys and credentials are never written to logs or included in error messages.
- All provider configuration and request payloads are validated by Pydantic before a request is dispatched.

### Known limitations

- Audio, text-to-speech, and transcription are **not implemented**. Every provider reports these capabilities as `False`, and requesting them raises `FeatureNotSupportedError`.
- The client is **synchronous only**; an async client is planned.
- The cache middleware is in-process and in-memory — it is not shared across workers or processes.

[Unreleased]: https://github.com/GenAIwithMS/uai-sdk-python/compare/v0.1.1...HEAD
[0.1.5]: https://github.com/GenAIwithMS/uai-sdk-python/compare/v0.1.1...v0.1.5
[0.1.0]: https://github.com/GenAIwithMS/uai-sdk-python/releases/tag/v0.1.0
