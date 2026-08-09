"""
Offline benchmarking for chat-capable models across providers.

Iterates every chat-capable model in the provider registry (DeepSeek,
Qwen, GLM, Kimi, StepFun, Doubao, MiniMax, Hunyuan) and measures
latency, time-to-first-token, throughput, and estimated cost per model.

Usage (programmatic):

.. code-block:: python

    from uai.benchmark import benchmark_models

    results = benchmark_models(iterations=5)
    for r in results:
        print(r.provider, r.model, f"{r.latency_ms_avg:.0f}ms", r.error_rate)

The CLI wrapper is ``uai benchmark`` (see ``uai.cli``).
"""

from __future__ import annotations

import concurrent.futures
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from uai import UniversalAI
from uai.models import StreamChunk, UnifiedResponse
from uai.registry import get_provider_config, list_providers

DEFAULT_PROMPT = "Write a short paragraph about the history of computing."

DEFAULT_ITERATIONS = 10
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_TOKENS = 256


@dataclass
class SampleResult:
    """A single benchmark iteration outcome."""

    success: bool
    latency_ms: float | None = None
    ttft_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for one provider/model pair."""

    provider: str
    model: str
    iterations: int
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.samples if s.success)

    @property
    def failure_count(self) -> int:
        return self.iterations - self.success_count

    @property
    def error_rate(self) -> float:
        return self.failure_count / self.iterations if self.iterations else 0.0

    def _values(self, attr: str) -> list[float]:
        return [
            getattr(s, attr) for s in self.samples if s.success and getattr(s, attr) is not None
        ]

    @property
    def ttft_ms_avg(self) -> float | None:
        values = self._values("ttft_ms")
        return statistics.fmean(values) if values else None

    @property
    def ttft_ms_p95(self) -> float | None:
        """95th-percentile TTFT; falls back to the max for fewer than 5 samples."""
        values = self._values("ttft_ms")
        return (
            statistics.quantiles(values, n=20)[18]
            if len(values) >= 5
            else (max(values) if values else None)
        )

    @property
    def latency_ms_avg(self) -> float | None:
        values = self._values("latency_ms")
        return statistics.fmean(values) if values else None

    @property
    def latency_ms_p95(self) -> float | None:
        """95th-percentile latency; falls back to the max for fewer than 5 samples."""
        values = self._values("latency_ms")
        return (
            statistics.quantiles(values, n=20)[18]
            if len(values) >= 5
            else (max(values) if values else None)
        )

    @property
    def tokens_per_sec(self) -> float | None:
        total_output = sum(s.output_tokens for s in self.samples if s.success)
        total_seconds = sum((s.latency_ms or 0.0) for s in self.samples if s.success) / 1000.0
        if total_seconds <= 0:
            return None
        return total_output / total_seconds

    @property
    def cost_usd_total(self) -> float:
        return sum(s.cost_usd for s in self.samples)

    @property
    def input_tokens_total(self) -> int:
        return sum(s.input_tokens for s in self.samples)

    @property
    def output_tokens_total(self) -> int:
        return sum(s.output_tokens for s in self.samples)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4) used when usage is unavailable."""
    return max(1, len(text) // 4)


def _run_iteration(
    client: UniversalAI,
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> SampleResult:
    """Run a single chat call and measure latency / TTFT / cost."""
    start = time.monotonic()
    ttft_ms: float | None = None
    input_tokens = 0
    output_tokens = 0
    try:
        if stream:
            last: StreamChunk | None = None
            for chunk in client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                if chunk.content and ttft_ms is None:
                    ttft_ms = (time.monotonic() - start) * 1000
                last = chunk
            latency_ms = (time.monotonic() - start) * 1000
            if last is not None and last.usage is not None:
                input_tokens = last.usage.input_tokens
                output_tokens = last.usage.output_tokens
            else:
                # Provider did not report usage on the stream — estimate.
                input_tokens = _estimate_tokens(prompt)
        else:
            response: UnifiedResponse = client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = (time.monotonic() - start) * 1000
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            if response.content:
                output_tokens = output_tokens or _estimate_tokens(response.content)

        pricing = get_provider_config(provider).get_model(model).pricing
        cost = pricing.cost_for(input_tokens, output_tokens)
        return SampleResult(
            success=True,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
    except Exception as error:  # benchmark reports all failures
        return SampleResult(success=False, error=str(error))


def _benchmark_model(
    client: UniversalAI,
    provider: str,
    model: str,
    prompt: str,
    iterations: int,
    max_workers: int,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> BenchmarkResult:
    """Benchmark one model with a thread pool."""
    samples: list[SampleResult] = []
    workers = max(1, min(max_workers, iterations))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_iteration,
                client,
                provider,
                model,
                prompt,
                max_tokens,
                temperature,
                stream,
            )
            for _ in range(iterations)
        ]
        for future in concurrent.futures.as_completed(futures):
            samples.append(future.result())
    return BenchmarkResult(
        provider=provider,
        model=model,
        iterations=iterations,
        samples=samples,
    )


def benchmark_models(
    providers: list[str] | None = None,
    models: list[str] | None = None,
    prompt: str = DEFAULT_PROMPT,
    iterations: int = DEFAULT_ITERATIONS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    stream: bool = True,
    client_factory: Callable[[str], UniversalAI] | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark chat-capable models across one or more providers.

    Args:
        providers: Provider names to benchmark (default: all registered).
        models: Model ids to benchmark per provider (default: all chat
            models of each provider).
        prompt: Prompt text to send.
        iterations: Number of calls per model.
        max_workers: Parallel worker count.
        max_tokens: Max output tokens per call.
        temperature: Sampling temperature.
        stream: Use streaming to measure time-to-first-token.
        client_factory: Override client creation (mainly for tests).

    Returns:
        One :class:`BenchmarkResult` per benchmarked model. Non-chat models
        are skipped.
    """
    provider_names = providers or list_providers()
    make_client = client_factory or (lambda p: UniversalAI(provider=p))
    results: list[BenchmarkResult] = []

    for provider in provider_names:
        config = get_provider_config(provider)
        model_ids = models or list(config.models.keys())
        for model_id in model_ids:
            # A global --models filter must not crash on providers that do
            # not offer the requested model; skip it instead.
            if model_id not in config.all_model_ids:
                continue
            model_info = config.get_model(model_id)
            if not model_info.capabilities.chat:
                continue
            use_stream = stream and model_info.capabilities.streaming
            client = make_client(provider)
            result = _benchmark_model(
                client,
                provider,
                model_id,
                prompt,
                iterations,
                max_workers,
                max_tokens,
                temperature,
                use_stream,
            )
            results.append(result)
    return results
