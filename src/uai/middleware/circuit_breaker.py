"""
Circuit breaker middleware (Module 1.4.2).

Detects sustained failures for a provider/model pair and fast-fails
subsequent requests until the provider recovers, so a degraded provider
cannot cause wasted network calls, throttling, or latency spikes.

States per (provider, model) key:

* ``closed`` — normal operation. Failures are counted; once the count
  reaches ``failure_threshold`` the circuit opens.
* ``open`` — requests are rejected immediately (``UAICircuitOpenError``,
  or a configured fallback response via :class:`MiddlewareHalt`) without
  touching the network. After ``reset_timeout`` seconds the circuit moves
  to ``half_open``.
* ``half_open`` — a single trial request is allowed through. If it
  succeeds the circuit closes (counters reset); if it fails the circuit
  reopens for another ``reset_timeout``.

Because the client is synchronous, only one half-open probe can be
in flight at a time, which is exactly the desired probe semantics.

Failures observed here are those that escape the execute chain before a
successful response is produced — the same boundary RetryMiddleware uses,
so a circuit breaker composes naturally with it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from uai.exceptions import UAICircuitOpenError
from uai.middleware.base import BaseMiddleware, MiddlewareContext
from uai.middleware.engine import MiddlewareHalt

logger = logging.getLogger(__name__)


class CircuitBreakerMiddleware(BaseMiddleware):
    """
    Fast-fail requests for a provider/model after repeated failures.

    Register this *before* (outside of) ``RetryMiddleware`` so an open
    circuit short-circuits immediately instead of consuming retries:

    .. code-block:: python

        from uai.middleware import CircuitBreakerMiddleware, RetryMiddleware

        client.use(CircuitBreakerMiddleware(failure_threshold=5, reset_timeout=30.0))
        client.use(RetryMiddleware(max_retries=3))

    Args:
        failure_threshold: Consecutive failures before the circuit opens
            (default 5).
        reset_timeout: Seconds the circuit stays open before allowing a
            half-open probe (default 30.0).
        fallback_response: Optional response served (via
            :class:`MiddlewareHalt`) when the circuit is open, instead of
            raising :class:`UAICircuitOpenError`.
        logger: Optional logger (defaults to ``uai.middleware``).
    """

    name = "circuit_breaker"

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        fallback_response: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout must be > 0")
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.fallback_response = fallback_response
        self._logger = logger or logging.getLogger("uai.middleware.circuit_breaker")

        # Per (provider, model) key state.
        self._failures: dict[tuple[str, str], int] = {}
        self._open_since: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Observability / management
    # ------------------------------------------------------------------

    @staticmethod
    def _key(provider: str | None, model: str | None) -> tuple[str, str]:
        return (provider or "", model or "")

    def state(self, provider: str | None, model: str | None) -> str:
        """
        Current circuit state (``"closed"``, ``"open"``, or ``"half_open"``)
        for a provider/model pair.
        """
        key = self._key(provider, model)
        opened = self._open_since.get(key)
        if opened is None:
            return "closed"
        if time.monotonic() - opened >= self.reset_timeout:
            return "half_open"
        return "open"

    def failures(self, provider: str | None, model: str | None) -> int:
        """Consecutive failure count for a provider/model pair."""
        return self._failures.get(self._key(provider, model), 0)

    def reset(self, provider: str | None = None, model: str | None = None) -> None:
        """
        Reset the circuit to ``closed`` — everything, a single provider, or
        a single provider/model pair.
        """
        if provider is None and model is None:
            self._failures.clear()
            self._open_since.clear()
            return
        if model is not None:
            key = self._key(provider, model)
            self._failures.pop(key, None)
            self._open_since.pop(key, None)
            return
        # Provider-only: reset every model for that provider.
        self._failures = {k: v for k, v in self._failures.items() if k[0] != provider}
        self._open_since = {k: v for k, v in self._open_since.items() if k[0] != provider}

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

    def execute(self, call_next: Callable[[], Any], context: MiddlewareContext) -> Any:
        """Guard the chain with the circuit state machine."""
        provider = context.provider
        model = context.model
        key = self._key(provider, model)

        if self.state(provider, model) in ("open", "half_open"):
            self._guard_open_circuit(key, provider, model)

        if context.request is not None and context.request.stream:
            return self._execute_stream(call_next, key, provider, model)
        return self._execute(call_next, key, provider, model)

    def _execute(
        self,
        call_next: Callable[[], Any],
        key: tuple[str, str],
        provider: str | None,
        model: str | None,
    ) -> Any:
        """Guard a non-streaming call."""
        try:
            response = call_next()
        except MiddlewareHalt:
            # A deliberate halt from an inner middleware is not a provider
            # failure — pass it through without counting it.
            raise
        except Exception:  # middleware boundary
            self._record_failure(key, provider, model)
            raise
        else:
            self._record_success(key, provider, model)
            return response

    def _execute_stream(
        self,
        call_next: Callable[[], Any],
        key: tuple[str, str],
        provider: str | None,
        model: str | None,
    ) -> Any:
        """
        Guard a streaming call by pulling the first chunk synchronously.

        A generator is created lazily, so without pulling the first chunk a
        stream that fails before delivering anything would be recorded as a
        success.  Pulling the first chunk makes the observed failure
        boundary match RetryMiddleware (pre-first-chunk), while mid-stream
        failures remain invisible to the breaker (the same trade-off retry
        makes).
        """
        try:
            generator = call_next()
            first = next(generator)
        except StopIteration:
            # Provider returned an empty stream; nothing to account for.
            return iter(())
        except MiddlewareHalt:
            raise
        except Exception:  # middleware boundary
            self._record_failure(key, provider, model)
            raise
        else:
            self._record_success(key, provider, model)

        def _generator():
            yield first
            yield from generator

        return _generator()

    def _guard_open_circuit(
        self, key: tuple[str, str], provider: str | None, model: str | None
    ) -> None:
        """
        When the circuit is open/half_open: fast-fail while open, or allow
        the single half-open probe through.
        """
        elapsed = time.monotonic() - self._open_since.get(key, time.monotonic())
        if elapsed < self.reset_timeout:
            # Circuit is open — reject without touching the network.
            self._logger.warning("[uai] circuit open %s/%s — rejecting request", provider, model)
            if self.fallback_response is not None:
                raise MiddlewareHalt(self.fallback_response)
            raise UAICircuitOpenError(
                f"Circuit open for provider '{provider}' (model '{model}')",
                provider=provider,
                model=model,
            )
        # Reset timeout elapsed — allow a single half-open probe through.
        self._logger.info("[uai] circuit half-open %s/%s — probing", provider, model)

    def _record_failure(
        self, key: tuple[str, str], provider: str | None, model: str | None
    ) -> None:
        """Count a failure; open (or reopen) the circuit when warranted."""
        opened = self._open_since.get(key)
        if opened is not None and time.monotonic() - opened >= self.reset_timeout:
            # Half-open probe failed — reopen for another reset_timeout and
            # start counting from a clean slate.
            self._failures.pop(key, None)
            self._open_since[key] = time.monotonic()
            self._logger.warning("[uai] circuit reopened %s/%s", provider, model)
            return

        failures = self._failures.get(key, 0) + 1
        self._failures[key] = failures
        if failures >= self.failure_threshold:
            self._open_since[key] = time.monotonic()
            self._logger.warning(
                "[uai] circuit opened %s/%s after %d failures", provider, model, failures
            )

    def _record_success(
        self, key: tuple[str, str], provider: str | None, model: str | None
    ) -> None:
        """A success resets counters; a successful probe closes the circuit."""
        self._failures.pop(key, None)
        if key in self._open_since:
            self._open_since.pop(key, None)
            self._logger.info("[uai] circuit closed %s/%s", provider, model)


__all__ = ["CircuitBreakerMiddleware"]
