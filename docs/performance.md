# Performance Targets & Systemic Testing

The SDK commits to concrete non-functional guarantees (Module 1.6) and
validates them with an automated KPI regression suite that runs against an
in-process **mock provider server** — no credentials or external network
required. This keeps performance from silently regressing across releases.

## KPI Targets

| KPI | Target | Where it's validated |
|-----|--------|----------------------|
| Internal processing overhead per request | **< 5 ms** | `tests/performance/test_kpi_overhead.py` |
| Streaming TTFT handling overhead | **< 30 ms** | `tests/performance/test_kpi_overhead.py` |
| Throughput | **≥ 1,000 req/min** | `tests/performance/test_kpi_throughput.py` |
| Idle memory footprint (SDK loaded) | **< 50 MB marginal** (target < 30 MB) | `tests/performance/test_kpi_memory.py` |
| Memory under ~100 parallel requests | **< 150 MB** | `tests/performance/test_kpi_memory.py` |

> **Note on idle memory:** the dependency baseline alone (pydantic v2 +
> httpx) is ~28 MB on modern CPython, so the idle KPI is asserted at a
> regression-safe 50 MB *marginal* (SDK over a bare interpreter) and can be
> tightened via `UAI_PERF_IDLE_MB`. The sustained-load cap keeps the plan's
> 150 MB target.

## Running the KPI Suite

```bash
# All non-memory KPIs (fast: overhead, TTFT, throughput)
poetry run pytest tests/performance -v

# Include the memory footprint KPIs (subprocess-based, ~10s)
UAI_PERF_MEMORY=1 poetry run pytest tests/performance -v
```

The KPI suite runs automatically in CI (`tests/performance` job) with
`UAI_PERF_MEMORY=1`. Thresholds can be tuned per environment without code
changes:

| Env var | Default | Meaning |
|---------|---------|---------|
| `UAI_PERF_MEMORY` | (unset) | `1` enables the memory KPIs |
| `UAI_PERF_IDLE_MB` | `50` | Idle marginal footprint cap |
| `UAI_PERF_SUSTAINED_MB` | `150` | Sustained-load peak cap |

## The Mock Provider Server

`uai.testing.MockProviderServer` is a **dependency-free** (stdlib-only)
in-process HTTP server that speaks the OpenAI-compatible subset of the
provider wire API, so the full client → middleware → adapter → network →
parse pipeline can be exercised offline:

```python
from uai import UniversalAI
from uai.testing import MockProviderServer

with MockProviderServer(latency_ms=5.0) as server:
    import os
    os.environ["UAI_PROVIDER_DEEPSEEK_BASE_URL"] = server.base_url
    client = UniversalAI(api_key="test", provider="deepseek")

    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    for chunk in client.chat(messages=[{"role": "user", "content": "Hi"}], stream=True):
        print(chunk.content)
```

Features:

- **Endpoints** — `POST /chat/completions` (JSON and SSE streaming when
  `stream: true`), `POST /embeddings`, with 404 for unknown paths.
- **`latency_ms` / `chunk_delay_ms`** — simulate network and streaming
  latency to make timing tests deterministic.
- **`fail_with(status, count)`** — inject transient failures (e.g. 429)
  then recover, to exercise retry/backoff offline.
- **`request_count`** — counts served requests for throughput assertions.
- Thread-safe (`ThreadingHTTPServer`), usable as a context manager, and
  binds an ephemeral port by default.

## Resource Optimizations

Module 1.6.1 also drives genuine footprint reductions:

- **Lazy adapter loading** — provider adapters are imported on first use
  instead of at `import uai`, so an application that only ever touches
  one provider never pays the import cost of the other seven adapter
  modules (faster cold start, smaller loaded module set).
- The metric aggregation engine and tracing middleware remain dependency-
  free and add sub-millisecond overhead per call.
