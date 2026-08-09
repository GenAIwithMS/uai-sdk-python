"""
Logging middleware.

Emits structured log lines for each request, response, and final error,
correlated by ``request_id``. Secrets are never logged — only operation
name, provider, model, message count, latency, usage, and finish reason.
"""

from __future__ import annotations

import logging
from typing import Any

from uai.middleware.base import BaseMiddleware, MiddlewareContext


class LoggingMiddleware(BaseMiddleware):
    """
    Log request/response lifecycle events.

    Args:
        logger: Optional logger (defaults to ``uai.middleware``).
        level: Logging level for request/response lines (default INFO).
    """

    name = "logging"

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        self._logger = logger or logging.getLogger("uai.middleware")
        self._level = level

    def before_request(
        self,
        request: Any,
        context: MiddlewareContext,
    ) -> Any:
        n_messages = len(request.messages) if request is not None else 0
        stream = request.stream if request is not None else None
        self._logger.log(
            self._level,
            "[uai] %s request provider=%s model=%s stream=%s messages=%d request_id=%s",
            context.operation,
            context.provider,
            context.model,
            stream,
            n_messages,
            context.request_id,
        )
        return request

    def after_response(
        self,
        response: Any,
        context: MiddlewareContext,
    ) -> Any:
        usage = getattr(response, "usage", None)
        finish_reason = getattr(response, "finish_reason", None)
        self._logger.log(
            self._level,
            "[uai] %s response provider=%s model=%s latency_ms=%.1f cache_hit=%s "
            "input_tokens=%s output_tokens=%s finish_reason=%s request_id=%s",
            context.operation,
            context.provider,
            context.model,
            context.elapsed_ms,
            context.cache_hit,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            finish_reason.value if finish_reason is not None else None,
            context.request_id,
        )
        return response

    def on_error(self, error: BaseException, context: MiddlewareContext) -> None:
        self._logger.warning(
            "[uai] %s failed provider=%s model=%s error=%r request_id=%s",
            context.operation,
            context.provider,
            context.model,
            error,
            context.request_id,
        )
