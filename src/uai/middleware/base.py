"""
Base classes for the UAI middleware pipeline.

Middleware are opt-in interceptors that wrap the request/response lifecycle
of the SDK client. They are registered explicitly via ``client.use(...)``,
so the default path (no middleware) stays fast and simple.

The pipeline follows the interceptor pattern described in the
implementation plan (Module 1.4):

1. ``before_request`` hooks run in registration order — each may mutate
   the :class:`~uai.models.UnifiedRequest`.
2. The request is executed through the ``execute`` chain — each middleware
   may wrap the next step (used by retry and cache).
3. ``after_response`` hooks run in reverse registration order — each may
   mutate the response.
4. ``on_error`` hooks run (in reverse order) when the chain raises.

Middleware are **synchronous**, matching the synchronous client.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from uai.models import UnifiedRequest


@dataclass
class MiddlewareContext:
    """
    Per-request context shared with every middleware in the chain.

    Attributes:
        operation: The SDK operation name (``"chat"``, ``"embed"``, ``"rerank"``).
        provider: The provider that served the request.
        model: The model that served the request.
        request: The normalized ``UnifiedRequest`` (``None`` for embed/rerank).
        start_time: ``time.monotonic()`` value when the pipeline started.
        request_id: Short random id for correlating logs/spans.
        attempt: Current retry attempt (0 = first try).
        cache_hit: Whether a cache middleware served the response.
        error: The error raised by the chain, if any.
        span: Reserved for tracing middleware (e.g. a ``Span`` object).
    """

    operation: str
    provider: str | None = None
    model: str | None = None
    request: UnifiedRequest | None = None
    start_time: float = field(default_factory=time.monotonic)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    attempt: int = 0
    cache_hit: bool = False
    error: BaseException | None = None
    span: Any = None

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds elapsed since the pipeline started."""
        return (time.monotonic() - self.start_time) * 1000


class BaseMiddleware:
    """
    Base class for all middleware.

    Subclasses override any of the four hooks. The default implementations
    are pass-throughs, so a middleware can implement only what it needs.
    """

    name: str = "base"

    def before_request(
        self,
        request: UnifiedRequest | None,
        context: MiddlewareContext,
    ) -> UnifiedRequest | None:
        """
        Run before the request is executed (in registration order).

        May inspect or mutate the request. Return the (possibly modified)
        request.
        """
        return request

    def execute(
        self,
        call_next: Callable[[], Any],
        context: MiddlewareContext,
    ) -> Any:
        """
        Wrap the remainder of the chain.

        ``call_next()`` executes the rest of the middleware chain and,
        ultimately, the network call. Middleware that need to intercept
        the actual call (retry, cache) override this method.

        Args:
            call_next: Zero-argument callable that runs the rest of the chain.
            context: The request context.

        Returns:
            The response produced by the chain.
        """
        return call_next()

    def after_response(
        self,
        response: Any,
        context: MiddlewareContext,
    ) -> Any:
        """
        Run after the response is produced (in reverse registration order).

        May inspect or mutate the response. Return the (possibly modified)
        response.
        """
        return response

    def on_error(self, error: BaseException, context: MiddlewareContext) -> None:
        """
        Called (in reverse registration order) when the chain raises.

        Note: ``on_error`` fires only for errors that escape the chain —
        a retry middleware that eventually succeeds will never trigger it.
        """
