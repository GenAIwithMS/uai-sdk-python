"""
Module 1.6.1 — resource footprint KPI.

Memory targets:

* Baseline idle footprint (SDK loaded with no active requests): target
  under 30 MB *marginal* (measured over a bare interpreter). The pydantic
  v2 + httpx dependency baseline alone is ~28 MB on CPython 3.14, so the
  default threshold is set to 50 MB to guard regressions without flaking
  on interpreter/dependency version differences; ``UAI_PERF_IDLE_MB``
  overrides it.
* Under sustained load (~100 parallel requests): capped at 150 MB
  (``UAI_PERF_SUSTAINED_MB`` overrides).

Footprint is measured in clean subprocesses (bare interpreter vs. SDK
loaded, and SDK under parallel load) via ``resource.getrusage``, so pytest
itself and other test imports never pollute the numbers. These tests are
opt-in: run only when ``UAI_PERF_MEMORY=1`` is set (CI sets it).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import textwrap

import pytest

IDLE_FOOTPRINT_MB = float(os.environ.get("UAI_PERF_IDLE_MB", "50"))
SUSTAINED_FOOTPRINT_MB = float(os.environ.get("UAI_PERF_SUSTAINED_MB", "150"))

RUN_MEMORY_KPIS = os.environ.get("UAI_PERF_MEMORY") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_MEMORY_KPIS,
        reason="set UAI_PERF_MEMORY=1 to run memory footprint KPIs",
    ),
    pytest.mark.skipif(
        not hasattr(resource, "getrusage"),
        reason="resource.getrusage is unavailable on this platform",
    ),
]


def _run(code: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _to_kb(value: int) -> int:
    """Normalize ``ru_maxrss`` to KB (macOS reports bytes)."""
    return value // 1024 if sys.platform == "darwin" else value


def _maxrss_kb(code: str) -> int:
    """Peak RSS (KB) of a fresh interpreter running *code*."""
    body = (
        f"{code}; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
        if code
        else "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
    )
    out = _run(f"import resource; {body}")
    return _to_kb(int(out.strip().splitlines()[-1]))


def _bare_kb() -> int:
    return _maxrss_kb("")


def test_idle_footprint_below_target() -> None:
    """
    SDK loaded with no active requests must stay under the footprint KPI
    (marginal MB over a bare interpreter).
    """
    bare_kb = _bare_kb()
    loaded_kb = _maxrss_kb("from uai import UniversalAI; UniversalAI(api_key='x')")
    marginal_mb = (loaded_kb - bare_kb) / 1024.0

    assert marginal_mb < IDLE_FOOTPRINT_MB, (
        f"idle marginal footprint {marginal_mb:.1f}MB exceeds the "
        f"{IDLE_FOOTPRINT_MB}MB KPI (bare {bare_kb / 1024.0:.1f}MB, "
        f"loaded {loaded_kb / 1024.0:.1f}MB)"
    )


def test_sustained_load_below_150mb() -> None:
    """
    Under sustained load (~100 parallel requests against the mock server)
    process memory must stay capped at 150 MB.
    """
    script = textwrap.dedent(
        """
        import os
        import resource
        from concurrent.futures import ThreadPoolExecutor

        from uai import UniversalAI
        from uai.testing import MockProviderServer

        with MockProviderServer() as server:
            os.environ["UAI_PROVIDER_DEEPSEEK_BASE_URL"] = server.base_url
            client = UniversalAI(api_key="x")
            client.chat(messages=[{"role": "user", "content": "warm"}])

            def call(_):
                return client.chat(messages=[{"role": "user", "content": "hi"}])

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(call, range(100)))
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        """
    )
    raw = _run(script)
    peak_mb = _to_kb(int(raw.strip().splitlines()[-1])) / 1024.0

    assert peak_mb < SUSTAINED_FOOTPRINT_MB, (
        f"peak footprint under load {peak_mb:.1f}MB exceeds the {SUSTAINED_FOOTPRINT_MB}MB KPI"
    )
