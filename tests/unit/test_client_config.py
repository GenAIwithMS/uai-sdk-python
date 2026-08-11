"""Unit tests for the UniversalAI constructor's ``timeout`` and ``max_retries``.

Both were previously written to ``self._config``, an object the request paths
never read, so neither parameter had any effect. ``timeout`` now resolves at
the point of use and ``max_retries`` composes a RetryMiddleware innermost.
"""

from __future__ import annotations

import pytest

import uai.client as client_module
from uai import UniversalAI
from uai.exceptions import UAIRateLimitError
from uai.middleware import BaseMiddleware, CacheMiddleware, RetryMiddleware
from uai.registry import get_provider_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("UAI_PROVIDER_DEEPSEEK_TIMEOUT", raising=False)
    monkeypatch.delenv("UAI_PROVIDER_DEEPSEEK_MAX_RETRIES", raising=False)


def _ok_response():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "id-1", "choices": [{"message": {"content": "ok"}}], "usage": {}}

    return FakeResponse()


class TestTimeout:
    def test_constructor_timeout_reaches_the_request(self, monkeypatch):
        # Regression: the constructor value landed on self._config, which no
        # request path reads, so every call used the registry default of 30s.
        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["timeout"] = timeout
            return _ok_response()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(provider="deepseek", timeout=120.0)
        client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert captured["timeout"] == 120.0

    def test_registry_default_applies_when_unset(self):
        client = UniversalAI(provider="deepseek")
        assert client._timeout_for(get_provider_config("deepseek")) == 30.0

    def test_constructor_timeout_beats_env_override(self, monkeypatch):
        # Documented precedence: constructor arguments over environment.
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_TIMEOUT", "60")
        client = UniversalAI(provider="deepseek", timeout=120.0)
        assert client._timeout_for(client._config_for("deepseek")) == 120.0

    def test_env_override_applies_when_no_constructor_timeout(self, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_TIMEOUT", "60")
        client = UniversalAI(provider="deepseek")
        assert client._timeout_for(client._config_for("deepseek")) == 60.0

    def test_timeout_applies_across_providers(self):
        # Unlike a credential, a timeout expresses the caller's own deadline,
        # so it holds for a per-call provider= override too.
        client = UniversalAI(provider="deepseek", timeout=120.0)
        assert client._timeout_for(get_provider_config("qwen")) == 120.0

    def test_client_timeout_does_not_leak_into_other_clients(self):
        # _config_for hands back shared registry entries; the override must
        # never be written into them.
        UniversalAI(provider="deepseek", timeout=120.0)
        other = UniversalAI(provider="deepseek")

        assert other._timeout_for(get_provider_config("deepseek")) == 30.0
        assert get_provider_config("deepseek").timeout == 30.0


class TestMaxRetries:
    def test_max_retries_actually_retries(self, monkeypatch):
        # Regression: max_retries was accepted and silently discarded.
        attempts = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise UAIRateLimitError("429", provider="deepseek", status_code=429)
            return _ok_response()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(provider="deepseek", max_retries=3)
        client._auto_retry.base_delay = 0.0
        client._auto_retry.jitter = False

        assert client.chat(messages=[{"role": "user", "content": "Hi"}]).content == "ok"
        assert attempts["n"] == 3

    def test_retrying_stays_opt_in(self, monkeypatch):
        # The registry's max_retries default must not switch retrying on.
        attempts = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            attempts["n"] += 1
            raise UAIRateLimitError("429", provider="deepseek", status_code=429)

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(provider="deepseek")

        with pytest.raises(UAIRateLimitError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert attempts["n"] == 1
        assert client._auto_retry is None

    def test_exhausted_retries_raise_the_original_error(self, monkeypatch):
        attempts = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            attempts["n"] += 1
            raise UAIRateLimitError("429", provider="deepseek", status_code=429)

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(provider="deepseek", max_retries=2)
        client._auto_retry.base_delay = 0.0
        client._auto_retry.jitter = False

        with pytest.raises(UAIRateLimitError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert attempts["n"] == 3  # initial attempt + 2 retries

    @pytest.mark.parametrize("value", [None, 0])
    def test_no_retry_policy_for_unset_or_zero(self, value):
        client = UniversalAI(provider="deepseek", max_retries=value)
        assert client._auto_retry is None

    def test_explicit_retry_middleware_supersedes_the_shorthand(self, caplog):
        # Composing both would nest two retry loops and multiply attempts.
        client = UniversalAI(provider="deepseek", max_retries=5)
        client.use(RetryMiddleware(max_retries=2))

        assert client._auto_retry is None
        assert "max_retries=5 is ignored" in caplog.text

    def test_explicit_retry_middleware_in_a_list_is_detected(self):
        client = UniversalAI(provider="deepseek", max_retries=5)
        client.use([CacheMiddleware(), RetryMiddleware(max_retries=2)])
        assert client._auto_retry is None

    def test_unrelated_middleware_leaves_the_shorthand_intact(self):
        client = UniversalAI(provider="deepseek", max_retries=5)
        client.use(CacheMiddleware())
        assert client._auto_retry is not None

    def test_auto_retry_runs_inside_registered_middleware(self, monkeypatch):
        """Retry must be innermost, so outer middleware sees one final outcome."""
        attempts = {"n": 0}
        observed: list = []

        def fake_post(url, headers=None, json=None, timeout=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise UAIRateLimitError("429", provider="deepseek", status_code=429)
            return _ok_response()

        class Observer(BaseMiddleware):
            name = "observer"

            def before_request(self, request, context):
                observed.append("before")
                return request

            def after_response(self, response, context):
                observed.append("after")
                return response

            def on_error(self, error, context):
                observed.append("error")
                return None

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(provider="deepseek", max_retries=3)
        client._auto_retry.base_delay = 0.0
        client._auto_retry.jitter = False
        client.use(Observer())

        client.chat(messages=[{"role": "user", "content": "Hi"}])

        # Three network attempts, but the outer middleware ran once and never
        # saw the transient failures.
        assert attempts["n"] == 3
        assert observed == ["before", "after"]

    def test_streaming_retries_before_the_first_chunk(self, monkeypatch):
        attempts = {"n": 0}

        def fake_stream(*args, **kwargs):
            attempts["n"] += 1
            raise UAIRateLimitError("429", provider="deepseek", status_code=429)

        monkeypatch.setattr(client_module.httpx, "stream", fake_stream)
        client = UniversalAI(provider="deepseek", max_retries=2)
        client._auto_retry.base_delay = 0.0
        client._auto_retry.jitter = False

        with pytest.raises(UAIRateLimitError):
            list(client.chat(messages=[{"role": "user", "content": "Hi"}], stream=True))
        assert attempts["n"] == 3
