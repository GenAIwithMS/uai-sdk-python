"""
Metric aggregation engine and middleware (Module 1.5.1).

Records key operational metrics using standardized Prometheus naming
conventions (``uai_*``), collected in-process by a :class:`MetricsRegistry`
and renderable on demand in Prometheus text exposition format:

* ``uai_requests_total`` — counter, tagged by operation/provider/status
* ``uai_request_duration_seconds`` — histogram (end-to-end latency)
* ``uai_ttft_seconds`` — histogram (time-to-first-token, streaming)
* ``uai_tokens_input_total`` / ``uai_tokens_output_total`` — counters
* ``uai_cache_hits_total`` — counter
* ``uai_retries_total`` — counter
* ``uai_errors_total`` — counter, tagged by error type
* ``uai_provider_requests_total`` — counter, tagged by provider/status

Metrics are synchronous and dependency-free; if you run Prometheus, expose
``MetricsRegistry.render()`` from a metrics endpoint and scrape it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from uai.middleware.base import BaseMiddleware, MiddlewareContext

# Prometheus-friendly histogram buckets (seconds).
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _format_value(value: float) -> str:
    """Render a numeric value the way Prometheus expects (no trailing .0)."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


class MetricsRegistry:
    """
    In-process metric aggregation engine.

    Stores labeled counters and histograms keyed by metric name, and can
    render everything in Prometheus text exposition format via :meth:`render`.
    """

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.buckets = tuple(sorted(buckets))
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]], tuple[float, int, list[int]]
        ] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @staticmethod
    def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        """Normalize a label dict into a sortable, hashable key."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def increment(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        value: float = 1.0,
    ) -> None:
        """Increment a counter by *value* (default 1)."""
        key = (name, self._labels_key(labels))
        self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a single observation into a histogram."""
        key = (name, self._labels_key(labels))
        total, count, buckets = self._histograms.get(key, (0.0, 0, [0] * len(self.buckets)))
        total += value
        count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                buckets[i] += 1
        self._histograms[key] = (total, count, buckets)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Current value of a counter (0 if never incremented)."""
        return self._counters.get((name, self._labels_key(labels)), 0.0)

    def histogram_count(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Number of observations in a histogram."""
        return self._histograms.get((name, self._labels_key(labels)), (0.0, 0, []))[1]

    def histogram_sum(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Sum of observations in a histogram."""
        return self._histograms.get((name, self._labels_key(labels)), (0.0, 0, []))[0]

    def clear(self) -> None:
        """Drop all recorded metrics."""
        self._counters.clear()
        self._histograms.clear()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------    @staticmethod
    def _escape_label_value(value: str) -> str:
        """Escape backslashes and double-quotes in a label value."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        joined = ",".join(f'{k}="{MetricsRegistry._escape_label_value(v)}"' for k, v in labels)
        return "{" + joined + "}"

    def render(self) -> str:
        """
        Render all metrics in Prometheus text exposition format.

        Each metric name gets a single ``# TYPE`` line; Returns an empty
        string when nothing has been recorded.
        """
        lines: list[str] = []

        # Group series by metric name so the TYPE line appears exactly once.
        counters: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = defaultdict(list)
        for (name, labels), value in self._counters.items():
            counters[name].append((labels, value))
        for name in sorted(counters):
            lines.append(f"# TYPE {name} counter")
            for labels, value in counters[name]:
                lines.append(f"{name}{self._format_labels(labels)} {_format_value(value)}")

        histograms: dict[
            str, list[tuple[tuple[tuple[str, str], ...], tuple[float, int, list[int]]]]
        ] = defaultdict(list)
        for (name, labels), data in self._histograms.items():
            histograms[name].append((labels, data))
        for name in sorted(histograms):
            lines.append(f"# TYPE {name} histogram")
            for labels, (total, count, buckets) in histograms[name]:
                for i, bound in enumerate(self.buckets):
                    lines.append(
                        f"{name}_bucket"
                        f"{self._format_labels((*labels, ('le', _format_value(bound))))} "
                        f"{buckets[i]}"
                    )
                # Prometheus convention: the final bucket is always +Inf == count.
                lines.append(
                    f"{name}_bucket{self._format_labels((*labels, ('le', '+Inf')))} {count}"
                )
                lines.append(f"{name}_sum{self._format_labels(labels)} {_format_value(total)}")
                lines.append(f"{name}_count{self._format_labels(labels)} {count}")
        return "\n".join(lines) + ("\n" if lines else "")


class MetricsMiddleware(BaseMiddleware):
    """
    Record operational metrics for every operation passing through the chain.

    Args:
        registry: Optional :class:`MetricsRegistry` (a new one is created).
            Share one registry across clients to aggregate globally.
    """

    name = "metrics"

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or MetricsRegistry()

    def _base_labels(self, context: MiddlewareContext) -> dict[str, str]:
        labels = {"operation": context.operation}
        if context.provider:
            labels["provider"] = context.provider
        if context.model:
            labels["model"] = context.model
        return labels

    # -- Pipeline hooks ---------------------------------------------------

    def execute(self, call_next: Callable[[], Any], context: MiddlewareContext) -> Any:
        """Wrap the chain; for streaming, observe TTFT on the first content chunk."""
        if context.request is not None and context.request.stream:
            return self._execute_stream(call_next, context)
        return call_next()

    def _execute_stream(self, call_next: Callable[[], Any], context: MiddlewareContext) -> Any:
        generator = call_next()

        def _generator():
            for chunk in generator:
                ttft_ms = getattr(chunk, "ttft_ms", None)
                if ttft_ms is not None:
                    self.registry.observe(
                        "uai_ttft_seconds", ttft_ms / 1000.0, self._base_labels(context)
                    )
                yield chunk

        return _generator()

    def after_response(
        self,
        response: Any,
        context: MiddlewareContext,
    ) -> Any:
        """Record success-path metrics for a completed operation."""
        labels = self._base_labels(context)
        status_labels = dict(labels, status="success")

        self.registry.increment("uai_requests_total", status_labels)
        self.registry.increment(
            "uai_provider_requests_total",
            {"provider": context.provider or "", "status": "success"},
        )
        self.registry.observe("uai_request_duration_seconds", context.elapsed_ms / 1000.0, labels)
        if context.cache_hit:
            self.registry.increment("uai_cache_hits_total", {"operation": context.operation})
        if context.attempt:
            self.registry.increment("uai_retries_total", labels, value=float(context.attempt))

        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            if input_tokens:
                self.registry.increment("uai_tokens_input_total", labels, value=float(input_tokens))
            if output_tokens:
                self.registry.increment(
                    "uai_tokens_output_total", labels, value=float(output_tokens)
                )
        return response

    def on_error(self, error: BaseException, context: MiddlewareContext) -> None:
        """Record failure-path metrics when the chain raises."""
        labels = self._base_labels(context)
        self.registry.increment("uai_requests_total", dict(labels, status="error"))
        self.registry.increment(
            "uai_provider_requests_total",
            {"provider": context.provider or "", "status": "error"},
        )
        self.registry.increment("uai_errors_total", dict(labels, type=type(error).__name__))


__all__ = ["MetricsMiddleware", "MetricsRegistry"]
