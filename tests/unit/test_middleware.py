"""Unit tests for the middleware pipeline and built-in middleware."""

from __future__ import annotations

import logging
import time

import pytest

import uai.client as client_module
from uai import UniversalAI
from uai.exceptions import UAIError, UAINetworkError, UAIRateLimitError
from uai.middleware import (
    CacheMiddleware,
    LoggingMiddleware,
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
        assert client._middleware == [mw]

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

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(Recorder())
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=make_request()
        )
        out = list(client._wrap_stream(iter(()), ctx))
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

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(Recorder())
        ctx = MiddlewareContext(
            operation="chat", provider="deepseek", model="deepseek-chat", request=make_request()
        )

        def gen():
            yield {"content": "a"}
            raise UAINetworkError("mid-stream")

        out = client._wrap_stream(gen(), ctx)
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
