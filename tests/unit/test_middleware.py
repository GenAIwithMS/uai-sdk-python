"""Unit tests for the middleware pipeline and built-in middleware."""

from __future__ import annotations

import logging
import time

import pytest

import uai.client as client_module
import uai.middleware.circuit_breaker as circuit_breaker_module
from uai import UniversalAI
from uai.exceptions import (
    UAICircuitOpenError,
    UAIError,
    UAINetworkError,
    UAIRateLimitError,
)
from uai.middleware import (
    CacheMiddleware,
    CircuitBreakerMiddleware,
    LoggingMiddleware,
    MiddlewareEngine,
    MiddlewareHalt,
    RetryMiddleware,
    SpanRecorder,
    TracingMiddleware,
)
from uai.middleware.base import BaseMiddleware, MiddlewareContext
from uai.models import (
    ChatMessage,
    FinishReason,
    Role,
    UnifiedRequest,
    UnifiedResponse,
    UsageMetrics,
)


def make_request(**overrides) -> UnifiedRequest:
    params: dict = {
        "model": "deepseek-chat",
        "messages": [ChatMessage(role=Role.USER, content="Hello")],
    }
    params.update(overrides)
    return UnifiedRequest(**params)


def fake_http_post(captured: list | None = None):
    """Return an httpx.post replacement that records calls and returns a response."""

    def fake_post(url, headers=None, json=None, timeout=None):
        if captured is not None:
            captured.append(url)

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "id": "id-1",
                    "choices": [{"message": {"content": "Hello there"}, "role": "assistant"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                }

        return FakeResponse()

    return fake_post


class RecordingMiddleware(BaseMiddleware):
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self._events = events

    def before_request(self, request, context):
        self._events.append(f"before:{self.name}")
        return request

    def after_response(self, response, context):
        self._events.append(f"after:{self.name}")
        return response


class TestClientPipeline:
    def test_use_registers_middleware(self):
        client = UniversalAI(api_key="k", provider="deepseek")
        mw = RecordingMiddleware("m1", [])
        assert client.use(mw) is client
        assert client._engine.middleware == [mw]

    def test_use_rejects_non_middleware(self):
        client = UniversalAI(api_key="k", provider="deepseek")
        with pytest.raises(TypeError):
            client.use("not-a-middleware")

    def test_before_in_order_after_in_reverse(self, monkeypatch):
        events: list[str] = []
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(RecordingMiddleware("m1", events))
        client.use(RecordingMiddleware("m2", events))
        monkeypatch.setattr(client_module.httpx, "post", fake_http_post())
        client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert events == ["before:m1", "before:m2", "after:m2", "after:m1"]

    def test_on_error_runs_in_reverse_order(self, monkeypatch):
        events: list[str] = []

        class ErrMiddleware(BaseMiddleware):
            def __init__(self, name, events):
                self.name = name
                self._events = events

            def on_error(self, error, context):
                self._events.append(self.name)

        def failing_post(*args, **kwargs):
            raise client_module.httpx.TransportError("connection refused")

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(ErrMiddleware("m1", events))
        client.use(ErrMiddleware("m2", events))
        monkeypatch.setattr(client_module.httpx, "post", failing_post)
        with pytest.raises(UAINetworkError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert events == ["m2", "m1"]

    def test_retry_via_client(self, monkeypatch):
        calls = {"n": 0}

        def flaky_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise client_module.httpx.TransportError("boom")
            return fake_http_post()(url, headers=headers, json=json, timeout=timeout)

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False))
        monkeypatch.setattr(client_module.httpx, "post", flaky_post)
        response = client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert response.content == "Hello there"
        assert calls["n"] == 3

    def test_cache_via_client_avoids_network(self, monkeypatch):
        captured: list[str] = []
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(CacheMiddleware(ttl=60))
        monkeypatch.setattr(client_module.httpx, "post", fake_http_post(captured))
        client.chat(messages=[{"role": "user", "content": "Hi"}])
        client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert len(captured) == 1

    def test_before_request_returned_copy_is_executed(self, monkeypatch):
        class ReplaceMiddleware(BaseMiddleware):
            def before_request(self, request, context):
                return request.model_copy(update={"max_tokens": 999})

        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["max_tokens"] = json.get("max_tokens")
            return fake_http_post()(url, headers=headers, json=json, timeout=timeout)

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(ReplaceMiddleware())
        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client.chat(messages=[{"role": "user", "content": "Hi"}], max_tokens=5)
        assert captured["max_tokens"] == 999

    def test_empty_stream_still_runs_after_response(self):
        events: list[tuple] = []

        class Recorder(BaseMiddleware):
            def after_response(self, response, context):
                events.append(("after", response))
                return response

        engine = MiddlewareEngine()
        engine.use(Recorder())
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=make_request()
        )
        out = list(engine._wrap_stream(iter(()), ctx))
        assert out == []
        assert events == [("after", None)]

    def test_midstream_error_runs_on_error_not_after_response(self):
        events: list[tuple] = []

        class Recorder(BaseMiddleware):
            def after_response(self, response, context):
                events.append(("after", response))
                return response

            def on_error(self, error, context):
                events.append(("error", error))

        engine = MiddlewareEngine()
        engine.use(Recorder())
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=make_request()
        )

        def gen():
            yield {"content": "a"}
            raise UAINetworkError("mid-stream")

        out = engine._wrap_stream(gen(), ctx)
        assert next(out) == {"content": "a"}
        with pytest.raises(UAINetworkError):
            next(out)
        assert [e[0] for e in events] == ["error"]

    def test_midstream_error_span_is_marked_error(self):
        recorder = SpanRecorder()
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(TracingMiddleware(recorder=recorder))
        req = make_request(stream=True)

        def stream_fn(_ctx):
            def gen():
                yield {"content": "a"}
                raise UAINetworkError("mid-stream")

            return gen()

        out = client._run_stream_pipeline("chat", "deepseek", "deepseek-chat", req, stream_fn)
        next(out)
        with pytest.raises(UAINetworkError):
            next(out)
        span = recorder.spans[0]
        assert span.status == "error"
        assert span.end_time is not None


class TestRetryMiddleware:
    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            if calls["n"] < 3:
                raise UAINetworkError("boom")
            return "ok"

        mw = RetryMiddleware(max_retries=5, base_delay=0.001, jitter=False)
        ctx = MiddlewareContext(operation="chat", request=make_request())
        assert mw.execute(call_next, ctx) == "ok"
        assert calls["n"] == 3

    def test_gives_up_after_max_retries(self):
        mw = RetryMiddleware(max_retries=2, base_delay=0.001, jitter=False)
        ctx = MiddlewareContext(operation="chat", request=make_request())

        def always_fail():
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            mw.execute(always_fail, ctx)

    def test_does_not_retry_non_retryable(self):
        calls = {"n": 0}
        mw = RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False)
        ctx = MiddlewareContext(operation="chat", request=make_request())

        def bad_request():
            calls["n"] += 1
            raise UAIError("bad request", status_code=400)

        with pytest.raises(UAIError):
            mw.execute(bad_request, ctx)
        assert calls["n"] == 1

    def test_rate_limit_is_retried(self):
        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            if calls["n"] < 2:
                raise UAIRateLimitError("slow down", retry_after=0.001)
            return "ok"

        mw = RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False)
        ctx = MiddlewareContext(operation="chat", request=make_request())
        assert mw.execute(call_next, ctx) == "ok"
        assert calls["n"] == 2

    def test_streaming_retry_before_first_chunk(self):
        attempts = {"n": 0}

        def stream_fn():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise UAINetworkError("boom")

            def gen():
                yield {"content": "a"}
                yield {"content": "b"}

            return gen()

        mw = RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False)
        ctx = MiddlewareContext(operation="chat", request=make_request(stream=True))
        result = list(mw.execute(stream_fn, ctx))
        assert [c["content"] for c in result] == ["a", "b"]
        assert attempts["n"] == 2

    def test_streaming_error_after_first_chunk_not_retried(self):
        attempts = {"n": 0}

        def stream_fn():
            attempts["n"] += 1

            def gen():
                yield {"content": "a"}
                raise UAINetworkError("mid-stream")

            return gen()

        mw = RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False)
        ctx = MiddlewareContext(operation="chat", request=make_request(stream=True))
        out = mw.execute(stream_fn, ctx)
        assert next(out)["content"] == "a"
        with pytest.raises(UAINetworkError):
            next(out)
        assert attempts["n"] == 1


class TestCacheMiddleware:
    def test_hit_skips_call_next(self):
        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            return {"content": "hello"}

        mw = CacheMiddleware(ttl=60)
        req = make_request()
        ctx1 = MiddlewareContext(operation="chat", request=req)
        ctx2 = MiddlewareContext(operation="chat", request=req)
        assert mw.execute(call_next, ctx1) == {"content": "hello"}
        assert mw.execute(call_next, ctx2) == {"content": "hello"}
        assert calls["n"] == 1
        assert ctx2.cache_hit is True

    def test_distinct_requests_not_shared(self):
        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            return "ok"

        mw = CacheMiddleware(ttl=60)
        ctx_a = MiddlewareContext(
            operation="chat", request=make_request(messages=[{"role": "user", "content": "a"}])
        )
        ctx_b = MiddlewareContext(
            operation="chat", request=make_request(messages=[{"role": "user", "content": "b"}])
        )
        mw.execute(call_next, ctx_a)
        mw.execute(call_next, ctx_b)
        assert calls["n"] == 2

    def test_ttl_expiry(self):
        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            return "ok"

        mw = CacheMiddleware(ttl=0.05)
        ctx1 = MiddlewareContext(operation="chat", request=make_request())
        ctx2 = MiddlewareContext(operation="chat", request=make_request())
        mw.execute(call_next, ctx1)
        time.sleep(0.1)
        mw.execute(call_next, ctx2)
        assert calls["n"] == 2

    def test_skips_streaming(self):
        mw = CacheMiddleware(ttl=60)
        ctx = MiddlewareContext(operation="chat", request=make_request(stream=True))
        assert mw.execute(lambda: "stream", ctx) == "stream"
        assert ctx.cache_hit is False

    def test_output_schema_partitions_cache_key(self):
        # Two requests with identical messages but different output_schema
        # classes must NOT share a cache entry (Module 1.3.2).
        from pydantic import BaseModel

        class SchemaA(BaseModel):
            a: str

        class SchemaB(BaseModel):
            b: int

        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            return "ok"

        mw = CacheMiddleware(ttl=60)
        ctx_a = MiddlewareContext(
            operation="chat",
            request=make_request(
                messages=[{"role": "user", "content": "x"}], output_schema=SchemaA
            ),
        )
        ctx_b = MiddlewareContext(
            operation="chat",
            request=make_request(
                messages=[{"role": "user", "content": "x"}], output_schema=SchemaB
            ),
        )
        mw.execute(call_next, ctx_a)
        mw.execute(call_next, ctx_b)
        assert calls["n"] == 2
        # Same schema twice -> cache hit.
        mw.execute(call_next, ctx_a)
        assert calls["n"] == 2

    def test_skips_without_request(self):
        mw = CacheMiddleware(ttl=60)
        ctx = MiddlewareContext(operation="embed", request=None)
        assert mw.execute(lambda: "x", ctx) == "x"

    def test_clear(self):
        mw = CacheMiddleware(ttl=60)
        ctx = MiddlewareContext(operation="chat", request=make_request())
        mw.execute(lambda: "ok", ctx)
        assert mw.size == 1
        mw.clear()
        assert mw.size == 0


class TestLoggingMiddleware:
    def test_logs_request_and_response(self, caplog):
        mw = LoggingMiddleware()
        req = make_request()
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=req
        )
        with caplog.at_level(logging.INFO, logger="uai.middleware"):
            mw.before_request(req, ctx)
            resp = UnifiedResponse(
                content="hi",
                model="deepseek-chat",
                finish_reason=FinishReason.STOP,
                usage=UsageMetrics(input_tokens=5, output_tokens=3),
            )
            mw.after_response(resp, ctx)
        assert "chat request provider=deepseek model=deepseek-chat" in caplog.text
        assert "chat response" in caplog.text
        assert "input_tokens=5" in caplog.text

    def test_logs_error(self, caplog):
        mw = LoggingMiddleware()
        ctx = MiddlewareContext(operation="chat", provider="deepseek", model="deepseek-chat")
        with caplog.at_level(logging.WARNING, logger="uai.middleware"):
            mw.on_error(UAINetworkError("boom"), ctx)
        assert "chat failed" in caplog.text
        assert "boom" in caplog.text


class TestTracingMiddleware:
    def test_records_span_with_attributes(self):
        recorder = SpanRecorder()
        mw = TracingMiddleware(recorder=recorder)
        req = make_request(temperature=0.7, max_tokens=50)
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=req
        )
        mw.before_request(req, ctx)
        resp = UnifiedResponse(
            content="hi",
            model="deepseek-chat",
            finish_reason=FinishReason.STOP,
            usage=UsageMetrics(input_tokens=5, output_tokens=3),
        )
        mw.after_response(resp, ctx)

        span = recorder.spans[0]
        assert span.attributes["gen_ai.operation.name"] == "chat"
        assert span.attributes["gen_ai.request.model"] == "deepseek-chat"
        assert span.attributes["gen_ai.request.temperature"] == 0.7
        assert span.attributes["gen_ai.request.max_tokens"] == 50
        assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
        assert span.attributes["gen_ai.usage.input_tokens"] == 5
        assert span.status == "ok"
        assert span.duration_ms >= 0

    def test_error_marks_span_failed(self):
        recorder = SpanRecorder()
        mw = TracingMiddleware(recorder=recorder)
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=make_request()
        )
        mw.before_request(make_request(), ctx)
        mw.on_error(UAINetworkError("boom"), ctx)
        span = recorder.spans[0]
        assert span.status == "error"
        assert "boom" in span.error

    def test_records_new_genai_attributes(self):
        recorder = SpanRecorder()
        mw = TracingMiddleware(recorder=recorder)
        req = make_request(temperature=0.7, max_tokens=50, top_p=0.9, stop=["END"])
        req.tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "get_time"}},
        ]
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=req
        )
        mw.before_request(req, ctx)
        resp = UnifiedResponse(
            id="resp-42",
            content="hi",
            model="deepseek-chat",
            finish_reason=FinishReason.STOP,
            usage=UsageMetrics(
                input_tokens=5,
                output_tokens=3,
                cache_read_tokens=10,
                cache_write_tokens=20,
            ),
        )
        mw.after_response(resp, ctx)

        span = recorder.spans[0]
        assert span.attributes["gen_ai.provider.name"] == "deepseek"
        assert span.attributes["gen_ai.request.top_p"] == 0.9
        assert span.attributes["gen_ai.request.stop"] == ["END"]
        assert span.attributes["gen_ai.request.tools"] == ["get_weather", "get_time"]
        assert span.attributes["gen_ai.response.id"] == "resp-42"
        assert span.attributes["gen_ai.usage.cache_read_input_tokens"] == 10
        assert span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 20

    def _make_fake_otel(self):
        class FakeOtelSpan:
            def __init__(self, name, attributes):
                self.name = name
                self.attributes = dict(attributes)
                self.ended = False
                self.status = None
                self.exception = None

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def set_status(self, status):
                self.status = status

            def record_exception(self, exc):
                self.exception = exc

            def end(self):
                self.ended = True

        class FakeTracer:
            def __init__(self):
                self.spans = []

            def start_span(self, name, attributes=None, **kwargs):
                span = FakeOtelSpan(name, attributes or {})
                self.spans.append(span)
                return span

        return FakeTracer()

    def test_use_otel_creates_discrete_span(self):
        tracer = self._make_fake_otel()
        recorder = SpanRecorder()
        mw = TracingMiddleware(recorder=recorder, use_otel=True, tracer=tracer)
        req = make_request(temperature=0.5)
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=req
        )
        mw.before_request(req, ctx)
        resp = UnifiedResponse(
            content="hi",
            model="deepseek-chat",
            finish_reason=FinishReason.STOP,
            usage=UsageMetrics(input_tokens=5, output_tokens=3),
        )
        mw.after_response(resp, ctx)

        # The in-process span is recorded as before...
        assert len(recorder.spans) == 1
        # ...and a discrete OpenTelemetry span was created, decorated, closed.
        assert len(tracer.spans) == 1
        otel_span = tracer.spans[0]
        assert otel_span.name == "chat"
        assert otel_span.attributes["gen_ai.operation.name"] == "chat"
        assert otel_span.attributes["gen_ai.request.temperature"] == 0.5
        assert otel_span.attributes["gen_ai.response.model"] == "deepseek-chat"
        assert otel_span.status == "ok"
        assert otel_span.ended is True

    def test_use_otel_error_records_exception_and_status(self):
        tracer = self._make_fake_otel()
        mw = TracingMiddleware(use_otel=True, tracer=tracer)
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=make_request()
        )
        mw.before_request(make_request(), ctx)
        mw.on_error(UAINetworkError("boom"), ctx)

        otel_span = tracer.spans[0]
        assert otel_span.ended is True
        assert otel_span.exception is not None
        assert "boom" in str(otel_span.exception)
        assert otel_span.status == "error"


class TestMiddlewareHalt:
    """Module 1.4.1 — middleware can halt the execution flow entirely."""

    def test_before_request_halt_skips_network(self, monkeypatch):
        calls = {"n": 0}

        class HaltMiddleware(BaseMiddleware):
            def before_request(self, request, context):
                calls["n"] += 1
                raise MiddlewareHalt(
                    UnifiedResponse(
                        content="halted",
                        provider=context.provider,
                        model=context.model,
                        finish_reason=FinishReason.STOP,
                    )
                )

        def boom(*args, **kwargs):
            raise AssertionError("network must not be reached")

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(HaltMiddleware())
        monkeypatch.setattr(client_module.httpx, "post", boom)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert result.content == "halted"
        assert calls["n"] == 1  # before_request ran once

    def test_halt_runs_after_response_but_not_on_error(self):
        events: list[str] = []
        seen_contexts: list[MiddlewareContext] = []

        class HaltMiddleware(BaseMiddleware):
            def before_request(self, request, context):
                raise MiddlewareHalt({"content": "halted"})

        class Recorder(BaseMiddleware):
            def after_response(self, response, context):
                events.append("after")
                seen_contexts.append(context)
                return response

            def on_error(self, error, context):
                events.append("error")

        engine = MiddlewareEngine()
        engine.use(HaltMiddleware())
        engine.use(Recorder())

        result = engine.run(
            "chat", "deepseek", "deepseek-chat", make_request(), lambda ctx: "never"
        )
        assert result == {"content": "halted"}
        assert events == ["after"]  # after_response runs; on_error does not
        assert seen_contexts[0].halted is True

    def test_execute_chain_halt_skips_remaining_chain(self):
        events: list[str] = []

        class HaltMiddleware(BaseMiddleware):
            def execute(self, call_next, context):
                events.append("halt")
                raise MiddlewareHalt("short-circuited")

        class OuterMiddleware(BaseMiddleware):
            def execute(self, call_next, context):
                events.append("outer")
                return call_next()

        engine = MiddlewareEngine()
        engine.use(OuterMiddleware())
        engine.use(HaltMiddleware())

        result = engine.run(
            "chat", "deepseek", "deepseek-chat", make_request(), lambda ctx: "never"
        )
        assert result == "short-circuited"
        # outer wrapped the chain, halt inside it short-circuited the network
        assert events == ["outer", "halt"]

    def test_streaming_halt_replaces_provider_stream(self):
        seen_contexts: list[MiddlewareContext] = []

        class HaltMiddleware(BaseMiddleware):
            def before_request(self, request, context):
                seen_contexts.append(context)
                raise MiddlewareHalt(["chunk-1", "chunk-2"])

        engine = MiddlewareEngine()
        engine.use(HaltMiddleware())

        result = list(
            engine.run_stream("chat", "deepseek", "deepseek-chat", make_request(), lambda ctx: None)
        )
        assert result == ["chunk-1", "chunk-2"]
        assert seen_contexts[0].halted is True

    def test_halt_not_retried_by_retry_middleware(self):
        attempts = {"n": 0}

        class HaltMiddleware(BaseMiddleware):
            def execute(self, call_next, context):
                attempts["n"] += 1
                raise MiddlewareHalt("halted")

        engine = MiddlewareEngine()
        engine.use(RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False))
        engine.use(HaltMiddleware())

        result = engine.run(
            "chat", "deepseek", "deepseek-chat", make_request(), lambda ctx: "never"
        )
        assert result == "halted"
        assert attempts["n"] == 1  # halt is not retryable

    def test_streaming_execute_chain_halt(self):
        class HaltMiddleware(BaseMiddleware):
            def execute(self, call_next, context):
                raise MiddlewareHalt(["a", "b"])

        engine = MiddlewareEngine()
        engine.use(HaltMiddleware())

        result = list(
            engine.run_stream("chat", "deepseek", "deepseek-chat", make_request(), lambda ctx: None)
        )
        assert result == ["a", "b"]

    def test_non_halt_before_request_error_propagates_without_on_error(self):
        events: list[str] = []

        class BoomMiddleware(BaseMiddleware):
            def before_request(self, request, context):
                raise UAINetworkError("boom in before")

        class Recorder(BaseMiddleware):
            def on_error(self, error, context):
                events.append("error")

        engine = MiddlewareEngine()
        engine.use(BoomMiddleware())
        engine.use(Recorder())

        with pytest.raises(UAINetworkError):
            engine.run("chat", "deepseek", "deepseek-chat", make_request(), lambda ctx: "never")
        assert events == []  # before_request errors do not invoke on_error (pre-existing semantics)

    def test_halt_exported_from_package(self):
        from uai import MiddlewareHalt as TopLevelHalt
        from uai.middleware import MiddlewareHalt

        assert TopLevelHalt is MiddlewareHalt
        exc = MiddlewareHalt({"content": "x"})
        assert exc.response == {"content": "x"}

    def test_halt_through_client_chat(self, monkeypatch):
        class HaltMiddleware(BaseMiddleware):
            def before_request(self, request, context):
                raise MiddlewareHalt(
                    UnifiedResponse(
                        content="cached-fallback",
                        provider=context.provider,
                        model=context.model,
                        finish_reason=FinishReason.STOP,
                    )
                )

        def boom(*args, **kwargs):
            raise AssertionError("network must not be reached")

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(HaltMiddleware())
        monkeypatch.setattr(client_module.httpx, "post", boom)
        result = client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert result.content == "cached-fallback"


class TestCircuitBreakerMiddleware:
    """Module 1.4.2 — fast-fail a degraded provider until it recovers."""

    def _ctx(self, provider="deepseek", model="deepseek-chat"):
        return MiddlewareContext(
            operation="chat", provider=provider, model=model, request=make_request()
        )

    def test_counts_failures_below_threshold(self):
        cb = CircuitBreakerMiddleware(failure_threshold=3, reset_timeout=30)
        ctx = self._ctx()

        def fail():
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)
        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)
        assert cb.state("deepseek", "deepseek-chat") == "closed"
        assert cb.failures("deepseek", "deepseek-chat") == 2

        # A success resets the counter.
        assert cb.execute(lambda: "ok", ctx) == "ok"
        assert cb.failures("deepseek", "deepseek-chat") == 0

    def test_opens_after_threshold_and_rejects_without_network(self):
        cb = CircuitBreakerMiddleware(failure_threshold=2, reset_timeout=30)
        ctx = self._ctx()

        def fail():
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)
        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)
        assert cb.state("deepseek", "deepseek-chat") == "open"

        calls = {"n": 0}

        def never():
            calls["n"] += 1
            return "ok"

        with pytest.raises(UAICircuitOpenError):
            cb.execute(never, ctx)
        assert calls["n"] == 0  # call_next never invoked while open

    def test_half_open_probe_success_closes_circuit(self, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(circuit_breaker_module.time, "monotonic", lambda: clock["now"])

        cb = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30)
        ctx = self._ctx()

        def fail():
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)  # opens the circuit
        assert cb.state("deepseek", "deepseek-chat") == "open"

        # After reset_timeout elapses the circuit is half-open: one probe allowed.
        clock["now"] += 31.0
        assert cb.state("deepseek", "deepseek-chat") == "half_open"
        assert cb.execute(lambda: "ok", ctx) == "ok"  # probe succeeds
        assert cb.state("deepseek", "deepseek-chat") == "closed"

    def test_half_open_probe_failure_reopens(self, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(circuit_breaker_module.time, "monotonic", lambda: clock["now"])

        cb = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30)
        ctx = self._ctx()

        def fail():
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)  # opens
        clock["now"] += 31.0  # half-open
        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)  # probe fails -> reopened
        assert cb.state("deepseek", "deepseek-chat") == "open"
        assert cb.failures("deepseek", "deepseek-chat") == 0  # probe failures don't accumulate

    def test_fallback_response_halts_when_open(self):
        cb = CircuitBreakerMiddleware(
            failure_threshold=1,
            reset_timeout=30,
            fallback_response=UnifiedResponse(
                content="fallback", model="deepseek-chat", finish_reason=FinishReason.STOP
            ),
        )
        engine = MiddlewareEngine()
        engine.use(cb)

        def failing_execute(ctx):
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            engine.run("chat", "deepseek", "deepseek-chat", make_request(), failing_execute)

        # Circuit open -> request is halted with the fallback response, no error.
        events: list = []

        class Recorder(BaseMiddleware):
            def after_response(self, response, context):
                events.append(response)
                return response

        engine.use(Recorder())
        result = engine.run(
            "chat", "deepseek", "deepseek-chat", make_request(), lambda ctx: "never"
        )
        assert result.content == "fallback"
        assert events == [result]

    def test_per_key_isolation(self):
        cb = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30)

        def fail():
            raise UAINetworkError("boom")

        ctx_deepseek = self._ctx("deepseek", "deepseek-chat")
        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx_deepseek)
        assert cb.state("deepseek", "deepseek-chat") == "open"

        # A different provider/model is unaffected.
        ctx_qwen = self._ctx("qwen", "qwen-plus")
        assert cb.state("qwen", "qwen-plus") == "closed"
        assert cb.execute(lambda: "ok", ctx_qwen) == "ok"

    def test_open_error_not_retried_by_retry_middleware(self):
        engine = MiddlewareEngine()
        engine.use(RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False))
        engine.use(CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30))
        attempts = {"n": 0}

        def failing_execute(ctx):
            attempts["n"] += 1
            raise UAINetworkError("boom")

        with pytest.raises(UAICircuitOpenError):
            engine.run("chat", "deepseek", "deepseek-chat", make_request(), failing_execute)
        assert attempts["n"] == 1  # first call opens the circuit; retries are rejected fast

    def test_reset_closes_circuit(self):
        cb = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30)
        ctx = self._ctx()

        def fail():
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            cb.execute(fail, ctx)
        assert cb.state("deepseek", "deepseek-chat") == "open"

        cb.reset("deepseek", "deepseek-chat")
        assert cb.state("deepseek", "deepseek-chat") == "closed"
        assert cb.execute(lambda: "ok", ctx) == "ok"

    def test_reset_provider_only_resets_all_models(self):
        cb = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30)

        def fail():
            raise UAINetworkError("boom")

        with pytest.raises(UAINetworkError):
            cb.execute(fail, self._ctx("deepseek", "deepseek-chat"))
        with pytest.raises(UAINetworkError):
            cb.execute(fail, self._ctx("deepseek", "deepseek-reasoner"))
        with pytest.raises(UAINetworkError):
            cb.execute(fail, self._ctx("qwen", "qwen-plus"))

        cb.reset(provider="deepseek")
        assert cb.state("deepseek", "deepseek-chat") == "closed"
        assert cb.state("deepseek", "deepseek-reasoner") == "closed"
        assert cb.state("qwen", "qwen-plus") == "open"  # untouched

    def test_streaming_failure_before_first_chunk_trips_breaker(self):
        cb = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30)
        ctx = self._ctx()
        ctx.request = make_request(stream=True)

        def failing_stream():
            def gen():
                raise UAINetworkError("boom before first chunk")
                yield  # pragma: no cover

            return gen()

        with pytest.raises(UAINetworkError):
            cb.execute(failing_stream, ctx)
        assert cb.state("deepseek", "deepseek-chat") == "open"

    def test_streaming_success_counts_success(self):
        cb = CircuitBreakerMiddleware(failure_threshold=3, reset_timeout=30)
        ctx = self._ctx()
        ctx.request = make_request(stream=True)

        def ok_stream():
            def gen():
                yield {"content": "a"}
                yield {"content": "b"}

            return gen()

        result = list(cb.execute(ok_stream, ctx))
        assert [c["content"] for c in result] == ["a", "b"]
        assert cb.state("deepseek", "deepseek-chat") == "closed"
        assert cb.failures("deepseek", "deepseek-chat") == 0

    def test_mid_stream_failure_not_observed(self):
        # Matches RetryMiddleware's pre-first-chunk boundary: once the first
        # chunk is delivered, later failures are the caller's concern.
        cb = CircuitBreakerMiddleware(failure_threshold=1, reset_timeout=30)
        ctx = self._ctx()
        ctx.request = make_request(stream=True)

        def mid_stream_fail():
            def gen():
                yield {"content": "a"}
                raise UAINetworkError("mid-stream")

            return gen()

        out = cb.execute(mid_stream_fail, ctx)
        assert next(out)["content"] == "a"
        with pytest.raises(UAINetworkError):
            next(out)
        assert cb.state("deepseek", "deepseek-chat") == "closed"
        assert cb.failures("deepseek", "deepseek-chat") == 0

    def test_breaker_through_client_chat(self, monkeypatch):
        calls = {"n": 0}

        def flaky_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise client_module.httpx.TransportError("boom")
            return fake_http_post()(url, headers=headers, json=json, timeout=timeout)

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(CircuitBreakerMiddleware(failure_threshold=2, reset_timeout=30))
        monkeypatch.setattr(client_module.httpx, "post", flaky_post)

        with pytest.raises(UAINetworkError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])
        with pytest.raises(UAINetworkError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])

        # Circuit open: the third call is rejected without hitting the network.
        calls["n"] = 0
        with pytest.raises(UAICircuitOpenError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert calls["n"] == 0
