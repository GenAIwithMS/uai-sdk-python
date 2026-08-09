"""
Retry middleware with exponential backoff and jitter.

Retries transient failures — rate limits (``UAIRateLimitError``),
network-level failures (``UAINetworkError``), timeouts
(``UAITimeoutError``), and 5xx server errors — up to ``max_retries``
times. Non-retryable errors (authentication, 4xx) are re-raised
immediately.

For streaming requests, retries only happen if the failure occurs before
the first chunk is delivered (i.e. before any partial output reaches the
caller); once streaming has started, errors are re-raised as-is.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

from uai.exceptions import (
    ResponseParsingError,
    UAIError,
    UAINetworkError,
    UAIRateLimitError,
    UAITimeoutError,
)
from uai.middleware.base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)


class RetryMiddleware(BaseMiddleware):
    """
    Retry transient failures with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Base delay in seconds for the first retry (default 0.5).
        max_delay: Cap on the backoff delay in seconds (default 10.0).
        jitter: Add randomized jitter to each delay (default True).
        retry_on_status: HTTP status codes that are considered retryable.
        retry_on_parsing_error: When True, also retry
            :class:`ResponseParsingError` (structured-output validation
            failures).  Off by default — enable explicitly when you want
            the middleware to re-ask the model after malformed JSON.
        logger: Optional logger (defaults to ``uai.middleware``).
    """

    name = "retry"

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        jitter: bool = True,
        retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
        retry_on_parsing_error: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on_status = retry_on_status
        self.retry_on_parsing_error = retry_on_parsing_error
        self._logger = logger or logging.getLogger("uai.middleware.retry")

    # -- Helpers ----------------------------------------------------------

    def _should_retry(self, error: BaseException) -> bool:
        """Return True when *error* is a transient, retryable failure."""
        if isinstance(error, (UAIRateLimitError, UAINetworkError, UAITimeoutError)):
            return True
        if isinstance(error, UAIError) and error.status_code in self.retry_on_status:
            return True
        if self.retry_on_parsing_error and isinstance(error, ResponseParsingError):
            return True
        return False

    def _delay(self, attempt: int, error: BaseException) -> float:
        """Compute the backoff delay for *attempt* (1-based)."""
        base: float = self.base_delay
        if isinstance(error, UAIRateLimitError) and error.retry_after is not None:
            base = error.retry_after
        delay = min(self.max_delay, base * (2 ** (attempt - 1)))
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay

    # -- Pipeline hooks ---------------------------------------------------

    def execute(self, call_next: Callable[[], Any], context: MiddlewareContext) -> Any:
        """Wrap the chain with the retry loop."""
        if context.request is not None and context.request.stream:
            return self._execute_stream(call_next, context)
        return self._execute(call_next, context)

    def _execute(self, call_next: Callable[[], Any], context: MiddlewareContext) -> Any:
        attempts = 0
        while True:
            context.attempt = attempts
            try:
                return call_next()
            except Exception as error:  # - middleware boundary
                context.error = error
                if attempts >= self.max_retries or not self._should_retry(error):
                    raise
                attempts += 1
                delay = self._delay(attempts, error)
                self._logger.warning(
                    "[uai] retry %s %s/%s attempt %d/%d after error in %.2fs: %s",
                    context.operation,
                    context.provider,
                    context.model,
                    attempts,
                    self.max_retries,
                    delay,
                    error,
                )
                time.sleep(delay)

    def _execute_stream(self, call_next: Callable[[], Any], context: MiddlewareContext) -> Any:
        """Retry a streaming call until the first chunk is delivered."""
        attempts = 0
        while True:
            context.attempt = attempts
            try:
                generator = call_next()
                first = next(generator)
                break
            except StopIteration:
                # Provider returned an empty stream; nothing to retry.
                return iter(())
            except Exception as error:  # - middleware boundary
                context.error = error
                if attempts >= self.max_retries or not self._should_retry(error):
                    raise
                attempts += 1
                delay = self._delay(attempts, error)
                self._logger.warning(
                    "[uai] retry %s (stream) %s/%s attempt %d/%d in %.2fs: %s",
                    context.operation,
                    context.provider,
                    context.model,
                    attempts,
                    self.max_retries,
                    delay,
                    error,
                )
                time.sleep(delay)

        def _generator():
            yield first
            yield from generator

        return _generator()
