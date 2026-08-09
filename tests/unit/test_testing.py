"""
Unit tests for the MockProviderServer testing utility (Module 1.6).

These tests point a real ``UniversalAI`` client at the in-process mock
server (via the ``UAI_PROVIDER_*_BASE_URL`` env override) and exercise the
full client -> middleware -> adapter -> network -> parse pipeline offline.
"""

from __future__ import annotations

import math
import time

import pytest

from uai import UniversalAI
from uai.exceptions import UAIRateLimitError
from uai.testing import MockProviderServer


@pytest.fixture(scope="module")
def mock_server() -> MockProviderServer:
    with MockProviderServer() as server:
        yield server


@pytest.fixture()
def deepseek_client(mock_server: MockProviderServer, monkeypatch) -> UniversalAI:
    monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_BASE_URL", mock_server.base_url)
    return UniversalAI(api_key="sk-test", provider="deepseek")


class TestMockServer:
    def test_chat_non_streaming(self, deepseek_client: UniversalAI):
        response = deepseek_client.chat(messages=[{"role": "user", "content": "hello"}])
        assert response.content and "mock completion" in response.content
        assert response.provider == "deepseek"
        assert response.finish_reason.value == "stop"
        assert response.usage.input_tokens == 16
        assert response.usage.output_tokens > 0

    def test_chat_streaming(self, deepseek_client: UniversalAI):
        chunks = list(
            deepseek_client.chat(messages=[{"role": "user", "content": "hello"}], stream=True)
        )
        assert len(chunks) > 1
        content = "".join(c.content or "" for c in chunks)
        assert "mock completion" in content
        # First content chunk carries a TTFT measurement.
        ttft_values = [c.ttft_ms for c in chunks if c.ttft_ms is not None]
        assert ttft_values, "no chunk recorded a TTFT timestamp"
        assert chunks[-1].is_final

    def test_embeddings(self, mock_server: MockProviderServer, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_QWEN_BASE_URL", mock_server.base_url)
        client = UniversalAI(api_key="sk-test", provider="qwen")
        result = client.embed(["alpha", "beta"], model="text-embedding-v4")
        assert len(result.vectors) == 2
        for i, vector in enumerate(result.vectors):
            assert vector.dimension == mock_server.embed_dim
            expected = [
                round(math.sin((i + 1) * (j + 1)) * 10, 4) for j in range(mock_server.embed_dim)
            ]
            assert vector.values == expected

    def test_request_counting(self, deepseek_client: UniversalAI, mock_server: MockProviderServer):
        before = mock_server.request_count
        deepseek_client.chat(messages=[{"role": "user", "content": "a"}])
        deepseek_client.chat(messages=[{"role": "user", "content": "b"}])
        assert mock_server.request_count == before + 2

    def test_failure_injection_then_recovery(
        self, deepseek_client: UniversalAI, mock_server: MockProviderServer
    ):
        mock_server.fail_with(429, count=2)
        for _ in range(2):
            with pytest.raises(UAIRateLimitError):
                deepseek_client.chat(messages=[{"role": "user", "content": "hi"}])
        # Server recovered: the next request succeeds.
        response = deepseek_client.chat(messages=[{"role": "user", "content": "hi"}])
        assert response.content

    def test_latency_is_applied(self):
        with MockProviderServer(latency_ms=60.0) as server:
            start = time.perf_counter()
            from uai.client import httpx

            response = httpx.post(
                f"{server.base_url}/chat/completions",
                json={"model": "mock", "messages": []},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        assert elapsed_ms >= 50.0, f"latency not applied (elapsed={elapsed_ms:.1f}ms)"

    def test_chunk_delay_is_applied(self):
        """Each SSE chunk is delayed by ``chunk_delay_ms``."""
        from uai.testing.mock_server import _CHAT_COMPLETION_CONTENT

        word_count = len(_CHAT_COMPLETION_CONTENT.split(" "))
        with MockProviderServer(chunk_delay_ms=30.0) as server:
            from uai.client import httpx

            start = time.perf_counter()
            with httpx.stream(
                "POST",
                f"{server.base_url}/chat/completions",
                json={"model": "mock", "stream": True, "messages": []},
            ) as response:
                line_count = sum(1 for _ in response.iter_lines())
            elapsed_ms = (time.perf_counter() - start) * 1000
        # Each SSE event is emitted as (data line + blank line): one event
        # per word, plus the terminal chunk and [DONE].
        assert line_count == word_count * 2 + 4, f"unexpected SSE framing: {line_count} lines"
        # ~N words x 30ms chunk delay dominates the elapsed time.
        assert elapsed_ms >= 60.0, f"chunk delay not applied (elapsed={elapsed_ms:.1f}ms)"

    def test_failures_served_in_order(self, mock_server: MockProviderServer, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_BASE_URL", mock_server.base_url)
        mock_server.fail_with(429, count=1)
        mock_server.fail_with(500, count=1)
        client = UniversalAI(api_key="sk-test", provider="deepseek")

        from uai.exceptions import UAIError, UAIRateLimitError

        with pytest.raises(UAIRateLimitError):
            client.chat(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(UAIError):
            client.chat(messages=[{"role": "user", "content": "hi"}])
        # Recovered: the third call succeeds.
        response = client.chat(messages=[{"role": "user", "content": "hi"}])
        assert response.content

    def test_base_url_requires_running_server(self):
        server = MockProviderServer()
        with pytest.raises(RuntimeError):
            _ = server.base_url

    def test_restart_after_stop(self, mock_server: MockProviderServer):
        server = MockProviderServer()
        server.start()
        port = server.server_port
        server.stop()
        server.start()
        assert server.server_port == port  # same bind address reused
        server.stop()

    def test_unknown_path_returns_404(self, mock_server: MockProviderServer):
        from uai.client import httpx

        response = httpx.post(f"{mock_server.base_url}/nope", json={})
        assert response.status_code == 404
