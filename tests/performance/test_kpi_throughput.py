"""
Module 1.6.2 — throughput KPI.

The SDK processing architecture must sustain at least 1,000 requests per
minute against a provider with negligible CPU bottlenecking. Measured
against the loopback mock server with a bounded thread pool, which models
parallel application load while keeping the test hermetic.
"""

from __future__ import annotations

import concurrent.futures
import time

THROUGHPUT_TARGET_RPM = 1_000.0
REQUESTS = 120
WORKERS = 12


def test_throughput_exceeds_1000_rpm(make_perf_client) -> None:
    """
    Issue REQUESTS chat calls concurrently against the mock server and
    assert the achieved rate clears 1,000 requests/minute.
    """
    client = make_perf_client()
    messages = [{"role": "user", "content": "hello"}]

    # Warm-up a single call so connection pooling is settled.
    client.chat(messages=messages)

    start = time.perf_counter()

    def _call(_: int) -> None:
        response = client.chat(messages=messages)
        assert response.content is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(_call, range(REQUESTS)))

    elapsed_s = time.perf_counter() - start
    rpm = (REQUESTS / elapsed_s) * 60.0

    assert rpm >= THROUGHPUT_TARGET_RPM, (
        f"throughput {rpm:.0f} req/min below the {THROUGHPUT_TARGET_RPM:.0f} req/min KPI "
        f"({REQUESTS} requests in {elapsed_s:.2f}s)"
    )
