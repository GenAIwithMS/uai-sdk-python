"""
In-memory TTL cache middleware.

Caches non-streaming responses keyed by a hash of the normalized
``UnifiedRequest``, so identical requests are served from memory instead
of hitting the provider (and its billing). Streaming requests bypass the
cache because a cached *iterator* cannot be safely replayed.

The cache is a plain dict with per-entry expiry timestamps; on insert the
oldest entry is evicted once ``max_size`` is reached.
"""

from __future__ import annotations

import copy
import hashlib
import time
from typing import Any, Callable

from uai.middleware.base import BaseMiddleware, MiddlewareContext
from uai.models import UnifiedRequest


class CacheMiddleware(BaseMiddleware):
    """
    Cache non-streaming responses in memory with a TTL.

    Args:
        ttl: Cache lifetime in seconds (default 300).
        max_size: Maximum number of cached entries (default 1024).
        cache: Optional dict to use as the backing store (mainly for tests).
    """

    name = "cache"

    def __init__(self, ttl: float = 300.0, max_size: int = 1024, cache: dict | None = None) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._cache: dict[str, tuple[float, Any]] = cache if cache is not None else {}

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _key(request: UnifiedRequest) -> str:
        """Stable hash of the normalized request (schema-free, no output schema)."""
        payload = request.model_dump_json(
            exclude_none=True,
            exclude={"output_schema", "metadata"},
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get(self, key: str) -> Any | None:
        item = self._cache.get(key)
        if item is None:
            return None
        expires_at, response = item
        if time.monotonic() > expires_at:
            self._cache.pop(key, None)
            return None
        return copy.deepcopy(response)

    def _set(self, key: str, response: Any) -> None:
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        self._cache[key] = (time.monotonic() + self.ttl, copy.deepcopy(response))

    def _evict_oldest(self) -> None:
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
        self._cache.pop(oldest_key, None)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of currently cached entries."""
        return len(self._cache)

    # -- Pipeline hooks ---------------------------------------------------

    def execute(self, call_next: Callable[[], Any], context: MiddlewareContext) -> Any:
        """Serve from cache on hit; otherwise execute and store."""
        request = context.request
        if request is None or request.stream:
            return call_next()

        key = self._key(request)
        cached = self._get(key)
        if cached is not None:
            context.cache_hit = True
            return cached

        response = call_next()
        self._set(key, response)
        return response
