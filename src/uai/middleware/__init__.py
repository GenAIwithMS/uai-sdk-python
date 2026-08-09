"""
Middleware package for the Universal AI Provider SDK.

Opt-in interceptors that wrap the client request/response lifecycle.
Register them on a client with ``client.use(...)``:

.. code-block:: python

    from uai import UniversalAI
    from uai.middleware import RetryMiddleware, CacheMiddleware, LoggingMiddleware

    client = UniversalAI(provider="deepseek")
    client.use(LoggingMiddleware())
    client.use(RetryMiddleware(max_retries=3))
    client.use(CacheMiddleware(ttl=300))
"""

from uai.middleware.base import BaseMiddleware, MiddlewareContext
from uai.middleware.cache import CacheMiddleware
from uai.middleware.logging import LoggingMiddleware
from uai.middleware.retry import RetryMiddleware
from uai.middleware.tracing import Span, SpanRecorder, TracingMiddleware

__all__ = [
    "BaseMiddleware",
    "CacheMiddleware",
    "LoggingMiddleware",
    "MiddlewareContext",
    "RetryMiddleware",
    "Span",
    "SpanRecorder",
    "TracingMiddleware",
]
