"""
Shared fixtures for the Module 1.6 KPI regression suite.

The suite measures the SDK against a real in-process mock provider server
over loopback HTTP, so no credentials or external network are needed. The
``UAI_PROVIDER_DEEPSEEK_BASE_URL`` env override points the client at it.
The env var is restored to its prior value at session teardown so the
override never leaks into the unit-test suite.
"""

from __future__ import annotations

import os

import pytest

from uai import UniversalAI
from uai.testing import MockProviderServer


@pytest.fixture(scope="session")
def perf_server() -> MockProviderServer:
    """A mock provider with zero artificial latency, shared across KPI tests."""
    server = MockProviderServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture()
def _perf_env(perf_server: MockProviderServer) -> None:
    """Point the DeepSeek client config at the mock server for this session."""
    var = "UAI_PROVIDER_DEEPSEEK_BASE_URL"
    previous = os.environ.get(var)
    os.environ[var] = perf_server.base_url
    yield
    if previous is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = previous


@pytest.fixture()
def make_perf_client(_perf_env: None, perf_server: MockProviderServer):
    """Factory returning a fresh client pointed at the mock server."""

    def _make(**kwargs: object) -> UniversalAI:
        return UniversalAI(api_key="sk-perf", provider="deepseek", **kwargs)

    return _make
