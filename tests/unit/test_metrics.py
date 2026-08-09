"""Unit tests for the metric aggregation engine and middleware (Module 1.5.1)."""

from __future__ import annotations

import pytest

import uai.client as client_module
from uai import UniversalAI
from uai.exceptions import UAINetworkError
from uai.middleware import (
    CacheMiddleware,
    MetricsMiddleware,
    MetricsRegistry,
    MiddlewareEngine,
    RetryMiddleware,
)
from uai.middleware.base import MiddlewareContext
from uai.models import (
    ChatMessage,
    FinishReason,
    Role,
    StreamChunk,
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


def ctx(operation="chat", provider="deepseek", model="deepseek-chat", **overrides):
    params = dict(operation=operation, provider=provider, model=model, request=make_request())
    params.update(overrides)
    return MiddlewareContext(**params)


def base_labels(**extra) -> dict[str, str]:
    labels = {"operation": "chat", "provider": "deepseek", "model": "deepseek-chat"}
    labels.update(extra)
    return labels


def fake_http_post():
    """Return an httpx.post replacement that records calls and returns a response."""

    def fake_post(url, headers=None, json=None, timeout=None):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "id": "id-1",
                    "choices": [{"message": {"content": "Hello there"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                }

        return FakeResponse()

    return fake_post


class TestMetricsRegistry:
    def test_counter_increments_with_labels(self):
        reg = MetricsRegistry()
        reg.increment("uai_requests_total", {"operation": "chat", "status": "ok"})
        reg.increment("uai_requests_total", {"operation": "chat", "status": "ok"})
        reg.increment("uai_requests_total", {"operation": "chat", "status": "error"})
        assert reg.counter_value("uai_requests_total", {"operation": "chat", "status": "ok"}) == 2
        assert (
            reg.counter_value("uai_requests_total", {"operation": "chat", "status": "error"}) == 1
        )
        assert reg.counter_value("uai_requests_total", {"operation": "embed"}) == 0

    def test_counter_label_order_insensitive(self):
        reg = MetricsRegistry()
        reg.increment("x", {"a": "1", "b": "2"})
        assert reg.counter_value("x", {"b": "2", "a": "1"}) == 1

    def test_histogram_buckets_sum_count(self):
        reg = MetricsRegistry(buckets=(0.1, 0.5, 1.0))
        for value in (0.05, 0.2, 0.7, 1.5):
            reg.observe("uai_request_duration_seconds", value)
        assert reg.histogram_count("uai_request_duration_seconds") == 4
        assert reg.histogram_sum("uai_request_duration_seconds") == pytest.approx(2.45)
        rendered = reg.render()
        # Buckets are cumulative: 0.05<=0.1, 0.2<=0.5, 0.7<=1.0, 1.5<=+Inf
        assert 'uai_request_duration_seconds_bucket{le="0.1"} 1' in rendered
        assert 'uai_request_duration_seconds_bucket{le="0.5"} 2' in rendered
        assert 'uai_request_duration_seconds_bucket{le="1"} 3' in rendered
        assert 'uai_request_duration_seconds_bucket{le="+Inf"} 4' in rendered

    def test_render_prometheus_format(self):
        reg = MetricsRegistry()
        reg.increment("uai_requests_total", {"operation": "chat", "status": "ok"})
        reg.observe("uai_ttft_seconds", 0.25)
        text = reg.render()
        assert "# TYPE uai_requests_total counter" in text
        assert 'uai_requests_total{operation="chat",status="ok"} 1' in text
        assert "# TYPE uai_ttft_seconds histogram" in text
        assert "uai_ttft_seconds_sum 0.25" in text
        assert "uai_ttft_seconds_count 1" in text

    def test_render_empty(self):
        assert MetricsRegistry().render() == ""

    def test_render_type_line_once_per_metric(self):
        reg = MetricsRegistry()
        reg.increment("uai_requests_total", {"operation": "chat", "status": "ok"})
        reg.increment("uai_requests_total", {"operation": "embed", "status": "ok"})
        text = reg.render()
        assert text.count("# TYPE uai_requests_total counter") == 1
        assert 'uai_requests_total{operation="chat",status="ok"} 1' in text
        assert 'uai_requests_total{operation="embed",status="ok"} 1' in text

    def test_render_escapes_label_values(self):
        reg = MetricsRegistry()
        reg.increment("x", {"model": 'quote"back\\slash'})
        text = reg.render()
        assert 'model="quote\\"back\\\\slash"' in text

    def test_clear(self):
        reg = MetricsRegistry()
        reg.increment("x")
        reg.observe("y", 1.0)
        reg.clear()
        assert reg.counter_value("x") == 0
        assert reg.histogram_count("y") == 0


class TestMetricsMiddleware:
    def test_records_success_path_through_client(self, monkeypatch):
        registry = MetricsRegistry()
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(MetricsMiddleware(registry=registry))
        monkeypatch.setattr(client_module.httpx, "post", fake_http_post())

        client.chat(messages=[{"role": "user", "content": "Hi"}])

        labels = base_labels(status="success")
        assert registry.counter_value("uai_requests_total", labels) == 1
        assert (
            registry.counter_value(
                "uai_provider_requests_total", {"provider": "deepseek", "status": "success"}
            )
            == 1
        )
        assert registry.histogram_count("uai_request_duration_seconds", base_labels()) == 1
        assert registry.counter_value("uai_tokens_input_total", base_labels()) == 10
        assert registry.counter_value("uai_tokens_output_total", base_labels()) == 4
        assert registry.counter_value("uai_cache_hits_total", {"operation": "chat"}) == 0

    def test_records_errors(self, monkeypatch):
        registry = MetricsRegistry()
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(MetricsMiddleware(registry=registry))

        def failing_post(*args, **kwargs):
            raise client_module.httpx.TransportError("connection refused")

        monkeypatch.setattr(client_module.httpx, "post", failing_post)
        with pytest.raises(UAINetworkError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert registry.counter_value("uai_requests_total", base_labels(status="error")) == 1
        assert (
            registry.counter_value(
                "uai_errors_total",
                base_labels(type="UAINetworkError"),
            )
            == 1
        )
        assert (
            registry.counter_value(
                "uai_provider_requests_total", {"provider": "deepseek", "status": "error"}
            )
            == 1
        )

    def test_records_cache_hits(self, monkeypatch):
        registry = MetricsRegistry()
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(CacheMiddleware(ttl=60))
        client.use(MetricsMiddleware(registry=registry))
        monkeypatch.setattr(client_module.httpx, "post", fake_http_post())

        client.chat(messages=[{"role": "user", "content": "Hi"}])
        client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert registry.counter_value("uai_cache_hits_total", {"operation": "chat"}) == 1
        # Both calls count as requests; only the first hit the network.
        assert registry.counter_value("uai_requests_total", base_labels(status="success")) == 2

    def test_records_retries(self, monkeypatch):
        registry = MetricsRegistry()
        calls = {"n": 0}

        def flaky_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise client_module.httpx.TransportError("boom")
            return fake_http_post()(url, headers=headers, json=json, timeout=timeout)

        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False))
        client.use(MetricsMiddleware(registry=registry))
        monkeypatch.setattr(client_module.httpx, "post", flaky_post)

        client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert calls["n"] == 3
        assert registry.counter_value("uai_retries_total", base_labels()) == 2

    def test_records_streaming_ttft(self):
        registry = MetricsRegistry()
        mw = MetricsMiddleware(registry=registry)
        engine_ctx = ctx(operation="chat", provider="deepseek", model="deepseek-chat")
        engine_ctx.request = make_request(stream=True)

        def stream_fn(_ctx):
            def gen():
                yield StreamChunk(content="hi", ttft_ms=150.0)
                yield StreamChunk(is_final=True)

            return gen()

        out = mw.execute(lambda: stream_fn(None), engine_ctx)
        list(out)
        assert registry.histogram_count("uai_ttft_seconds", base_labels()) == 1
        assert registry.histogram_sum("uai_ttft_seconds", base_labels()) == pytest.approx(0.15)

    def test_records_streaming_duration_and_requests_through_engine(self):
        registry = MetricsRegistry()
        engine = MiddlewareEngine()
        engine.use(MetricsMiddleware(registry=registry))
        req = make_request(stream=True)

        def stream_fn(_ctx):
            def gen():
                yield StreamChunk(content="hi", ttft_ms=100.0)
                yield StreamChunk(
                    is_final=True, usage=UsageMetrics(input_tokens=3, output_tokens=1)
                )

            return gen()

        out = engine.run_stream("chat", "deepseek", "deepseek-chat", req, stream_fn)
        list(out)

        assert registry.histogram_count("uai_ttft_seconds", base_labels()) == 1
        assert registry.counter_value("uai_requests_total", base_labels(status="success")) == 1
        assert registry.histogram_count("uai_request_duration_seconds", base_labels()) == 1
        assert registry.counter_value("uai_tokens_input_total", base_labels()) == 3

    def test_records_through_engine_run(self):
        registry = MetricsRegistry()
        engine = MiddlewareEngine()
        engine.use(MetricsMiddleware(registry=registry))

        def execute_fn(_ctx):
            return UnifiedResponse(
                content="ok",
                provider="deepseek",
                model="deepseek-chat",
                finish_reason=FinishReason.STOP,
                usage=UsageMetrics(input_tokens=7, output_tokens=2),
            )

        engine.run("chat", "deepseek", "deepseek-chat", make_request(), execute_fn)
        assert registry.counter_value("uai_requests_total", base_labels(status="success")) == 1
        assert registry.histogram_count("uai_request_duration_seconds", base_labels()) == 1
        assert registry.counter_value("uai_tokens_input_total", base_labels()) == 7

    def test_metrics_exported_from_package(self):
        from uai import MetricsMiddleware as TopMetrics
        from uai.middleware import MetricsMiddleware, MetricsRegistry

        assert TopMetrics is MetricsMiddleware
        assert MetricsRegistry is not None
