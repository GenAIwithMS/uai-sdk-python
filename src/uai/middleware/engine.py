"""
Interceptor Execution Engine (Module 1.4.1).

The execution engine processes a registered chain of middleware classes:

1. ``before_request`` hooks run in registration order — each may mutate
   the :class:`~uai.models.UnifiedRequest`.
2. The request is executed through the ``execute`` chain — each
   middleware may wrap the next step (used by retry and cache).
3. ``after_response`` hooks run in reverse registration order — each may
   mutate the response.
4. ``on_error`` hooks run (in reverse order) when the chain raises.

The engine also lets individual middleware **halt the execution flow
entirely** based on runtime conditions: raise :class:`MiddlewareHalt` with
a response from any hook and the engine short-circuits — the execute
chain (and the network call) is skipped, and the supplied response is fed
through the ``after_response`` hooks.  ``on_error`` is *not* invoked for a
deliberate halt.

Middleware are **synchronous**, matching the synchronous client.
"""

from __future__ import annotations

from typing import Any, Callable

from uai.middleware.base import BaseMiddleware, MiddlewareContext


class MiddlewareHalt(Exception):
    """
    Raised by middleware to halt the execution flow and serve a response.

    A middleware can short-circuit the pipeline entirely — skipping the
    execute chain and any network call — by raising ``MiddlewareHalt``
    with the response it wants served:

    .. code-block:: python

        from uai.middleware.engine import MiddlewareHalt
        from uai.models import UnifiedResponse

        def before_request(self, request, context):
            if context.provider == "deepseek":
                raise MiddlewareHalt(
                    UnifiedResponse(content="fallback", model=context.model)
                )
            return request

    Attributes:
        response: The response to serve in place of the network call.
    """

    def __init__(self, response: Any) -> None:
        super().__init__("Middleware halted the execution flow")
        self.response = response


class MiddlewareEngine:
    """
    Executes a chain of registered middleware around a request.

    The client owns one engine and delegates ``use()`` plus the
    non-streaming/streaming pipeline runs to it.  Kept separate from the
    client so the interceptor mechanics are isolated and unit-testable
    (Module 1.4.1).
    """

    def __init__(self) -> None:
        self._middleware: list[BaseMiddleware] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def use(self, middleware: BaseMiddleware | list[BaseMiddleware]) -> MiddlewareEngine:
        """
        Register one or more middleware instances (opt-in pipeline).

        Hooks run in registration order for ``before_request`` and in
        reverse order for ``after_response``/``on_error``.

        Args:
            middleware: A single middleware or a list of middleware.

        Returns:
            The engine, for chaining.

        Raises:
            TypeError: If an item is not a :class:`BaseMiddleware`.
        """
        items = middleware if isinstance(middleware, (list, tuple)) else [middleware]
        for item in items:
            if not isinstance(item, BaseMiddleware):
                raise TypeError(f"Expected BaseMiddleware, got {type(item).__name__}")
            self._middleware.append(item)
        return self

    @property
    def middleware(self) -> list[BaseMiddleware]:
        """The registered middleware, in execution order."""
        return list(self._middleware)

    # ------------------------------------------------------------------
    # Chain composition
    # ------------------------------------------------------------------

    def _chain_call(
        self,
        execute_fn: Callable[[MiddlewareContext], Any],
        context: MiddlewareContext,
    ) -> Any:
        """Compose the middleware ``execute`` chain around *execute_fn*."""
        middleware = self._middleware

        def call(index: int) -> Any:
            if index >= len(middleware):
                # Read the request from the context so that before_request
                # hooks that return a *new* request object take effect.
                return execute_fn(context)
            return middleware[index].execute(lambda: call(index + 1), context)

        return call(0)

    # ------------------------------------------------------------------
    # Pipeline runs
    # ------------------------------------------------------------------

    def run(
        self,
        operation: str,
        provider: str,
        model: str,
        request: Any,
        execute_fn: Callable[[MiddlewareContext], Any],
    ) -> Any:
        """
        Run ``before -> execute -> after`` around a non-streaming callable.

        If a middleware raises :class:`MiddlewareHalt`, the execute chain
        is skipped and the halted response is served through the
        ``after_response`` hooks.
        """
        context = MiddlewareContext(
            operation=operation,
            provider=provider,
            model=model,
            request=request,
        )

        # ``_NOT_HALTED`` is a sentinel (not ``None``) so a middleware may
        # legitimately halt with a ``None`` response.
        halted: Any = _NOT_HALTED
        for mw in self._middleware:
            try:
                request = mw.before_request(request, context)
                context.request = request
            except MiddlewareHalt as exc:
                halted = exc.response
                context.halted = True
                context.error = None  # a halt is not an error
                break

        if halted is _NOT_HALTED:
            try:
                response = self._chain_call(execute_fn, context)
            except MiddlewareHalt as exc:
                halted = exc.response
                context.halted = True
                context.error = None
            except Exception as error:  # middleware boundary
                context.error = error
                for mw in reversed(self._middleware):
                    mw.on_error(error, context)
                raise

        if halted is not _NOT_HALTED:
            response = halted

        for mw in reversed(self._middleware):
            response = mw.after_response(response, context)
        return response

    def run_stream(
        self,
        operation: str,
        provider: str,
        model: str,
        request: Any,
        stream_fn: Callable[[MiddlewareContext], Any],
    ) -> Any:
        """
        Run ``before -> execute`` around a streaming callable; ``after``
        on finish.

        A :class:`MiddlewareHalt` raised in a ``before_request`` hook
        (or the execute chain) replaces the provider stream with the
        halted response; it is then wrapped so ``after_response`` still
        runs when the stream completes.
        """
        context = MiddlewareContext(
            operation=operation,
            provider=provider,
            model=model,
            request=request,
        )

        halted: Any = _NOT_HALTED
        for mw in self._middleware:
            try:
                request = mw.before_request(request, context)
                context.request = request
            except MiddlewareHalt as exc:
                halted = exc.response
                context.halted = True
                context.error = None  # a halt is not an error
                break

        if halted is _NOT_HALTED:
            try:
                generator = self._chain_call(stream_fn, context)
            except MiddlewareHalt as exc:
                halted = exc.response
                context.halted = True
                context.error = None

        if halted is not _NOT_HALTED:
            try:
                generator = iter(halted)
            except TypeError as exc:
                raise TypeError(
                    f"Streaming halt responses must be iterable (got {type(halted).__name__})"
                ) from exc

        return self._wrap_stream(generator, context)

    def _wrap_stream(self, generator: Any, context: MiddlewareContext) -> Any:
        """
        Wrap a stream generator so middleware hooks always run.

        ``after_response`` runs on clean completion (even for an empty
        stream, with ``None``); ``on_error`` runs on failure and
        ``after_response`` is skipped so the error status recorded by
        ``on_error`` is not overwritten.
        """
        last: Any = None
        failed = False
        try:
            for chunk in generator:
                last = chunk
                yield chunk
        except Exception as error:  # middleware boundary
            failed = True
            context.error = error
            for mw in reversed(self._middleware):
                mw.on_error(error, context)
            raise
        finally:
            if not failed:
                for mw in reversed(self._middleware):
                    last = mw.after_response(last, context)


class _NotHalted:
    """Sentinel distinguishing "no halt" from ``halted=None``."""


_NOT_HALTED = _NotHalted()

__all__ = ["MiddlewareEngine", "MiddlewareHalt"]
