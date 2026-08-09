"""
Module 1.6.1 / 1.6.2 — SDK internal overhead KPI.

The SDK core must add less than 5 ms of internal processing overhead per
request (request normalization + capability enforcement + middleware
pipeline + response parsing), and the streaming TTFT handling overhead
must stay under 30 ms.

These measurements intentionally stub the HTTP transport with instant
in-memory responses: the KPI is the SDK's *own* processing cost, and
network/connection-setup time (which varies wildly across CI runners)
must not leak into it. The mock-server end-to-end path is covered by the
throughput KPI test.
"""

from __future__ import annotations

import statistics
import time

import pytest

import uai.client as client_module
from uai.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    RetryMiddleware,
    TracingMiddleware,
)

OVERHEAD_TARGET_MS = 5.0
TTFT_HANDLING_TARGET_MS = 30.0

ITERATIONS = 200

_INSTANT_CHAT = {
    "id": "cmpl-instant",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "instant response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
}

_INSTANT_STREAM_LINES = [
    (
        'data: {"id":"cmpl-instant","object":"chat.completion.chunk","model":"deepseek-chat",'
        '"choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}'
    ),
    (
        'data: {"id":"cmpl-instant","object":"chat.completion.chunk","model":"deepseek-chat",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":2,"completion_tokens":2}}'
    ),
    "data: [DONE]",
]


class _InstantResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return _INSTANT_CHAT


class _InstantStream:
    status_code = 200
    request = None

    def iter_lines(self):
        return iter(_INSTANT_STREAM_LINES)

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture()
def stubbed_transport(monkeypatch) -> None:
    """Replace the module-level httpx transport with instant in-memory stubs."""

    def fake_post(url, headers=None, json=None, timeout=None):
        return _InstantResponse()

    def fake_stream(*args, **kwargs):
        return _InstantStream()

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    monkeypatch.setattr(client_module.httpx, "stream", fake_stream)


def test_sdk_overhead_below_5ms_per_request(make_perf_client, stubbed_transport) -> None:
    """
    Measure average SDK-internal processing time of a non-streaming chat
    call with an instant transport. Must average under 5 ms.
    """
    client = make_perf_client()
    client.use(RetryMiddleware(max_retries=1, base_delay=0.001, jitter=False))
    client.use(TracingMiddleware())
    client.use(MetricsMiddleware())
    client.use(LoggingMiddleware())

    messages = [{"role": "user", "content": "hello"}]

    # Warm-up: lazy imports, JIT/allocator/cache stabilization.
    for _ in range(10):
        client.chat(messages=messages)

    latencies_ms: list[float] = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        response = client.chat(messages=messages)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.content is not None

    avg_ms = statistics.fmean(latencies_ms)
    assert avg_ms < OVERHEAD_TARGET_MS, (
        f"SDK overhead {avg_ms:.2f}ms exceeds the {OVERHEAD_TARGET_MS}ms KPI "
        f"(p50={statistics.median(latencies_ms):.2f}ms)"
    )


def test_streaming_ttft_handling_below_30ms(make_perf_client, stubbed_transport) -> None:
    """
    Measure time from starting a streaming call to receiving the first
    content chunk through the SDK's stream handling (middleware, SSE
    parsing, chunk normalization). The transport yields its first chunk
    instantly, so the measured time is the SDK's TTFT handling overhead;
    it must stay under 30 ms.
    """
    client = make_perf_client()
    start = time.perf_counter()
    first_chunk_ms: float | None = None
    for chunk in client.chat(messages=[{"role": "user", "content": "hello"}], stream=True):
        if first_chunk_ms is None and chunk.content:
            first_chunk_ms = (time.perf_counter() - start) * 1000
            break

    assert first_chunk_ms is not None, "stream never yielded a content chunk"
    assert first_chunk_ms < TTFT_HANDLING_TARGET_MS, (
        f"TTFT handling {first_chunk_ms:.2f}ms exceeds the {TTFT_HANDLING_TARGET_MS}ms KPI"
    )
