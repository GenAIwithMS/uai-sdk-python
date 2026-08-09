# Universal AI Provider SDK — Documentation

## Welcome

This is the central documentation hub for the **Universal AI Provider SDK (uai-sdk-python)**.

## Getting Started

- [Architecture](architecture.md) — High-level design: middleware pipeline, provider adapter contract, `UnifiedRequest` lifecycle
- [Installation & Configuration](configuration.md) — API keys, env vars, YAML config, key rotation

## API Reference

- [Chat](chat.md) — Text-in/text-out conversational completions, history, system prompts
- [Structured Output](structured_output.md) — Pydantic-validated structured output
- [Streaming](streaming.md) — Server-Sent Events and TTFT
- [Tools](tools.md) — Function calling with OpenAI-style tool definitions
- [Embeddings](embeddings.md) — Text embedding operations, routed via provider adapters
- [Vision](vision.md) — Multi-modal image interpretation, routed via chat content blocks
- [Rerank](rerank.md) — Document ranking via provider adapters (Qwen, GLM)
- [Benchmark CLI](benchmark.md) — Offline benchmarking tool (`uai benchmark`)

> **Audio / voice / TTS / transcription** are **not** implemented yet. As
> previously decided, they are deferred (a large, MiniMax-focused lift); all
> providers report these capabilities as `False`. See [providers.md](providers.md).

## Developer Guide

- [Middleware](middleware.md) — Creating and composing interceptors (Cache, Retry, Circuit Breaker, Logging, Tracing, Metrics)
- [Benchmark](benchmark.md) — Offline benchmarking CLI covering all models
- [Telemetry](telemetry.md) — Prometheus-style metrics (in-process registry) and GenAI tracing spans
- [PDK](pdk.md) — Provider Development Kit: adding a new provider adapter
- [Providers](providers.md) — Capability matrix & provider-specific notes

## Project Links

- [Roadmap](../README.md#roadmap)
- [Contributing](../CONTRIBUTING.md)
- [Issue Labels](../.github/LABELS.md)
