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

## [0.1.1] — 2026-08-10

A correctness, documentation, and packaging-metadata release.

### Security

- **Fixed a credential leak across providers.** `UniversalAI._get_api_key()` returned the credential supplied to the constructor for *any* provider, ignoring which provider was actually being called. A client built as `UniversalAI(api_key="sk-deepseek", provider="deepseek")` that then issued `client.chat(..., provider="qwen")` transmitted the DeepSeek key to DashScope's API in the `Authorization` header — and likewise for `embed()` and `rerank()`, which accept the same `provider=` override.

  Constructor credentials are now **scoped to the client's default provider**. Any other provider resolves its own key from its `api_key_env_var`, and raises `ValueError` naming that variable when no key is available rather than falling back to an unrelated credential. Covered by regression tests in `tests/unit/test_client_credentials.py`.

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
[0.1.1]: https://github.com/GenAIwithMS/uai-sdk-python/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GenAIwithMS/uai-sdk-python/releases/tag/v0.1.0
