# Universal AI Provider SDK — Documentation

## Welcome

This is the central documentation hub for the **Universal AI Provider SDK (uai-sdk-python)**.

> **Note:** This doc index is a stub created during project scaffolding. Content for each sub-topic will be fleshed out as features are implemented according to the phased roadmap. See [README.md](../README.md#roadmap).

## Getting Started

- [Architecture](architecture.md) — High-level design: middleware pipeline, provider adapter contract, `UnifiedRequest` lifecycle
- [Installation & Configuration](configuration.md) — API keys, env vars, YAML config, key rotation

## API Reference

- [Chat](chat.md) — Text-in/text-out conversational completions, history, system prompts
- [Structured Output](structured_output.md) — JSON schema enforcement via Pydantic
- [Streaming](streaming.md) — Server-Sent Events, TTFT, chunk aggregation
- [Tools](tools.md) — Function calling & Model Context Protocol (MCP) integration
- [Embeddings](embeddings.md) — Text embedding operations (Phase 2)
- [Vision](vision.md) — Multi-modal image interpretation (Phase 2)
- [Rerank](rerank.md) — Document ranking (Phase 2)
- [Benchmark CLI](benchmark.md) — Offline benchmarking tool (Phase 2)

## Developer Guide

- [Middleware](middleware.md) — Creating and composing interceptors (Cache, Retry, Logging, Routing)
- [Telemetry](telemetry.md) — OpenTelemetry integration, Prometheus metric conventions
- [PDK](pdk.md) — Provider Development Kit: adding a new provider adapter
- [Providers](providers.md) — Capability matrix & provider-specific notes

## Project Links

- [Roadmap](../README.md#roadmap)
- [Contributing](../CONTRIBUTING.md)
- [Issue Labels](../.github/LABELS.md)
