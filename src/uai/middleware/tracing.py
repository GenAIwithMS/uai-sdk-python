"""
Tracing middleware.

Records one span per SDK operation with OpenTelemetry-compatible
"GenAI" semantic attributes:

- ``gen_ai.operation.name`` (``chat`` / ``embed`` / ``rerank``)
- ``gen_ai.request.model``, ``gen_ai.request.temperature``,
  ``gen_ai.request.max_tokens``
- ``gen_ai.response.model``, ``gen_ai.response.finish_reasons``
- ``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens``

Spans are collected in-process by a :class:`SpanRecorder` (inspect via
``recorder.spans``). If the ``opentelemetry`` packages are installed and
``use_otel=True`` is passed, attributes are also exported onto the current
OpenTelemetry span.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from uai.middleware.base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import opentelemetry  # noqa: F401

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


@dataclass
class Span:
    """A single recorded operation span."""

    name: str
    operation: str
    provider: str | None = None
    model: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """Span duration in milliseconds (ends at ``finish()`` time)."""
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        """Mark the span as finished."""
        self.end_time = time.monotonic()
        self.status = status
        self.error = error


class SpanRecorder:
    """In-process collector of finished operation spans."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def start(self, span: Span) -> Span:
        """Record a new span and return it."""
        self._spans.append(span)
        return span

    @property
    def spans(self) -> list[Span]:
        """All spans recorded so far."""
        return list(self._spans)

    def clear(self) -> None:
        """Drop all recorded spans."""
        self._spans.clear()


class TracingMiddleware(BaseMiddleware):
    """
    Record a span per operation with GenAI semantic attributes.

    Args:
        recorder: Optional :class:`SpanRecorder` (a new one is created).
        service_name: Service name reported with each span (default ``uai``).
        use_otel: Also export attributes onto the current OpenTelemetry
            span (requires the ``opentelemetry`` packages to be installed).
    """

    name = "tracing"

    def __init__(
        self,
        recorder: SpanRecorder | None = None,
        service_name: str = "uai",
        use_otel: bool = False,
    ) -> None:
        self.recorder = recorder or SpanRecorder()
        self.service_name = service_name
        if use_otel and not _OTEL_AVAILABLE:
            logger.warning(
                "[uai] TracingMiddleware(use_otel=True) but opentelemetry is not "
                "installed; recording spans in-process only."
            )
        self.use_otel = use_otel and _OTEL_AVAILABLE

    def before_request(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> Any:
        """Start a span for this operation."""
        span = self.recorder.start(
            Span(
                name=f"{context.operation}",
                operation=context.operation,
                provider=context.provider,
                model=context.model,
            )
        )
        span.set_attribute("gen_ai.operation.name", context.operation)
        span.set_attribute("uai.service.name", self.service_name)
        if context.provider:
            span.set_attribute("uai.request.provider", context.provider)
        if context.model:
            span.set_attribute("gen_ai.request.model", context.model)
        if request is not None:
            if request.temperature is not None:
                span.set_attribute("gen_ai.request.temperature", request.temperature)
            if request.max_tokens is not None:
                span.set_attribute("gen_ai.request.max_tokens", request.max_tokens)
            span.set_attribute("gen_ai.request.stream", request.stream)
        context.span = span
        return request

    def after_response(
        self,
        response: Any,
        context: MiddlewareContext,
    ) -> Any:
        """Finish the span and decorate it with response attributes."""
        span = context.span
        if span is None:
            return response

        if response is not None:
            model = getattr(response, "model", None)
            if model:
                span.set_attribute("gen_ai.response.model", str(model))
            finish_reason = getattr(response, "finish_reason", None)
            if finish_reason is not None:
                value = finish_reason.value if hasattr(finish_reason, "value") else finish_reason
                span.set_attribute("gen_ai.response.finish_reasons", [value])
            usage = getattr(response, "usage", None)
            if usage is not None:
                span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)

        span.set_attribute("uai.cache_hit", context.cache_hit)
        span.finish()
        logger.debug(
            "[uai] span %s provider=%s model=%s duration_ms=%.1f status=%s",
            context.operation,
            context.provider,
            context.model,
            span.duration_ms,
            span.status,
        )
        self._export_otel(span)
        return response

    def on_error(self, error: BaseException, context: MiddlewareContext) -> None:
        """Mark the span as failed when the chain raises."""
        span = context.span
        if span is None:
            return
        span.set_attribute("uai.error", str(error))
        span.finish(status="error", error=str(error))
        self._export_otel(span)

    def _export_otel(self, span: Span) -> None:  # pragma: no cover - optional dep
        """Copy span attributes onto the current OpenTelemetry span, if enabled."""
        if not self.use_otel:
            return
        try:
            from opentelemetry import trace as otel_trace

            otel_span = otel_trace.get_current_span()
            if otel_span.is_recording():
                for key, value in span.attributes.items():
                    otel_span.set_attribute(key, value)
                if span.status == "error":
                    otel_span.record_exception(Exception(span.error or "uai error"))
        except Exception:
            logger.debug("[uai] OpenTelemetry export skipped", exc_info=True)
