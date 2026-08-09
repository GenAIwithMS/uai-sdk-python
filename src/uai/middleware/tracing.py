"""
Tracing middleware (Module 1.5.2).

Every LLM invocation generates a **discrete span** annotated with
OpenTelemetry "GenAI" semantic attributes:

- ``gen_ai.operation.name`` (``chat`` / ``embed`` / ``rerank``)
- ``gen_ai.provider.name``, ``gen_ai.request.model``,
  ``gen_ai.request.temperature``, ``gen_ai.request.top_p``,
  ``gen_ai.request.max_tokens``, ``gen_ai.request.stop``,
  ``gen_ai.request.tools``
- ``gen_ai.response.model`` — the actual model that served the response,
  which may differ from the requested model (providers often route to
  date-stamped or fine-tuned variants)
- ``gen_ai.response.id``, ``gen_ai.response.finish_reasons``
- ``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens``,
  ``gen_ai.usage.cache_read_input_tokens``,
  ``gen_ai.usage.cache_creation_input_tokens``

Spans are always collected in-process by a :class:`SpanRecorder` (inspect
via ``recorder.spans``). When ``use_otel=True`` and the ``opentelemetry``
packages are installed, each invocation additionally creates a **discrete
distributed span** (kind ``CLIENT``) on the ``uai-sdk`` tracer — not just
attributes copied onto whatever span happens to be current — so the SDK
call appears as its own node in a distributed trace.

A ``tracer`` may be injected for testing or to use a custom tracer
without installing the opentelemetry packages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from uai.middleware.base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

_OTEL_TRACER_NAME = "uai-sdk"
# Mirrors ``uai.__version__``; kept local to avoid a circular import.
_SDK_VERSION = "0.1.0"

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
    Record a discrete span per operation with GenAI semantic attributes.

    Args:
        recorder: Optional :class:`SpanRecorder` (a new one is created).
        service_name: Service name reported with each span (default ``uai``).
        use_otel: Also create a discrete OpenTelemetry span per invocation
            (requires the ``opentelemetry`` packages, or an injected
            ``tracer``).
        tracer: Optional OpenTelemetry-compatible tracer. When provided,
            it is used instead of resolving the ``uai-sdk`` tracer from
            the installed ``opentelemetry`` packages (mainly for tests).
    """

    name = "tracing"

    def __init__(
        self,
        recorder: SpanRecorder | None = None,
        service_name: str = "uai",
        use_otel: bool = False,
        tracer: Any = None,
    ) -> None:
        self.recorder = recorder or SpanRecorder()
        self.service_name = service_name
        self._tracer = tracer
        self.use_otel = use_otel and (tracer is not None or _OTEL_AVAILABLE)
        if use_otel and not self.use_otel:
            logger.warning(
                "[uai] TracingMiddleware(use_otel=True) but opentelemetry is not "
                "installed and no tracer was injected; recording spans in-process only."
            )

    # -- Span lifecycle -----------------------------------------------------

    def _new_span(self, context: MiddlewareContext) -> Span:
        return Span(
            name=f"{context.operation}",
            operation=context.operation,
            provider=context.provider,
            model=context.model,
        )

    def _start_otel_span(self, span: Span) -> Any | None:
        """Create a discrete OpenTelemetry span, or None when disabled/failing."""
        if not self.use_otel:
            return None
        try:
            if self._tracer is not None:
                # Injected tracers choose their own kind (a real OTel tracer
                # expects the SpanKind enum, not our string sentinel).
                return self._tracer.start_span(span.name, attributes=dict(span.attributes))
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer(_OTEL_TRACER_NAME, _SDK_VERSION)
            return tracer.start_span(
                span.name,
                kind=otel_trace.SpanKind.CLIENT,
                attributes=dict(span.attributes),
            )
        except Exception:
            logger.debug("[uai] OpenTelemetry span creation skipped", exc_info=True)
            return None

    def _finish_otel_span(self, span: Span, otel_span: Any | None) -> None:
        """Copy attributes onto and close the OpenTelemetry span."""
        if otel_span is None:
            return
        try:
            for key, value in span.attributes.items():
                otel_span.set_attribute(key, value)
            if span.status == "error":
                otel_span.record_exception(Exception(span.error or "uai error"))
            if self._tracer is None:
                from opentelemetry import trace as otel_trace

                otel_span.set_status(
                    otel_trace.Status(
                        otel_trace.StatusCode.ERROR
                        if span.status == "error"
                        else otel_trace.StatusCode.OK,
                        str(span.error or "") if span.status == "error" else "",
                    )
                )
            else:
                # Injected tracers receive a simple status string; real OTel
                # tracers get a proper Status object above.
                otel_span.set_status("error" if span.status == "error" else "ok")
            otel_span.end()
        except Exception:
            logger.debug("[uai] OpenTelemetry span finish skipped", exc_info=True)

    # -- Pipeline hooks -----------------------------------------------------

    def before_request(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> Any:
        """Start a span for this operation and populate request attributes."""
        span = self._new_span(context)
        span.set_attribute("gen_ai.operation.name", context.operation)
        span.set_attribute("uai.service.name", self.service_name)
        if context.provider:
            span.set_attribute("gen_ai.provider.name", context.provider)
            span.set_attribute("uai.request.provider", context.provider)
        if context.model:
            span.set_attribute("gen_ai.request.model", context.model)
        if request is not None:
            if request.temperature is not None:
                span.set_attribute("gen_ai.request.temperature", request.temperature)
            if request.max_tokens is not None:
                span.set_attribute("gen_ai.request.max_tokens", request.max_tokens)
            if request.top_p is not None:
                span.set_attribute("gen_ai.request.top_p", request.top_p)
            if request.stop:
                stop = request.stop if isinstance(request.stop, list) else [request.stop]
                span.set_attribute("gen_ai.request.stop", stop)
            if request.tools:
                # Tools may be ToolDefinition models or raw dicts (the
                # client assigns dicts post-construction) — handle both.
                names: list[str] = []
                for tool in request.tools:
                    function = getattr(tool, "function", None)
                    if isinstance(function, dict):
                        name = function.get("name", "")
                    elif function is not None:
                        name = getattr(function, "name", "")
                    elif isinstance(tool, dict):
                        name = tool.get("function", {}).get("name", "")
                    else:
                        name = getattr(tool, "name", "")
                    if name:
                        names.append(name)
                span.set_attribute("gen_ai.request.tools", names)
            span.set_attribute("gen_ai.request.stream", request.stream)
        self.recorder.start(span)
        context.span = span
        context.otel_span = self._start_otel_span(span)
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
            response_id = getattr(response, "id", None)
            if response_id:
                span.set_attribute("gen_ai.response.id", str(response_id))
            finish_reason = getattr(response, "finish_reason", None)
            if finish_reason is not None:
                value = finish_reason.value if hasattr(finish_reason, "value") else finish_reason
                span.set_attribute("gen_ai.response.finish_reasons", [value])
            usage = getattr(response, "usage", None)
            if usage is not None:
                span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
                if usage.cache_read_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.cache_read_input_tokens", usage.cache_read_tokens
                    )
                if usage.cache_write_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.cache_creation_input_tokens", usage.cache_write_tokens
                    )

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
        self._finish_otel_span(span, context.otel_span)
        return response

    def on_error(self, error: BaseException, context: MiddlewareContext) -> None:
        """Mark the span as failed when the chain raises."""
        span = context.span
        if span is None:
            return
        span.set_attribute("uai.error", str(error))
        span.finish(status="error", error=str(error))
        self._finish_otel_span(span, context.otel_span)


__all__ = ["Span", "SpanRecorder", "TracingMiddleware"]
