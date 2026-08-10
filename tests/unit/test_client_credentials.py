"""Unit tests for credential scoping in the UniversalAI client.

Constructor credentials belong to the client's default provider and must
never be reused for a different provider reached through a per-call
``provider=`` override — doing so would transmit one provider's API key to
another provider's API.
"""

from __future__ import annotations

import pytest

import uai.client as client_module
from uai import UniversalAI
from uai.registry import get_provider_config

# Every provider API key env var, so a developer's real environment cannot
# leak into these assertions.
_API_KEY_ENV_VARS = [
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "BIGMODEL_API_KEY",
    "MOONSHOT_API_KEY",
    "STEPFUN_API_KEY",
    "DOUBAO_API_KEY",
    "MINIMAX_API_KEY",
    "HUNYUAN_API_KEY",
]


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    """Run every test against a clean credential environment."""
    for var in _API_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestCredentialScoping:
    def test_constructor_key_serves_the_default_provider(self):
        client = UniversalAI(api_key="sk-deepseek", provider="deepseek")
        assert client._get_api_key(get_provider_config("deepseek")) == "sk-deepseek"

    @pytest.mark.parametrize("other", ["qwen", "glm", "kimi", "hunyuan"])
    def test_constructor_key_does_not_leak_to_other_providers(self, other):
        # Regression: _get_api_key returned the constructor credential for
        # *any* provider, so client.chat(provider="qwen") sent the DeepSeek
        # key to DashScope.
        client = UniversalAI(api_key="sk-deepseek", provider="deepseek")
        assert client._get_api_key(get_provider_config(other)) is None

    def test_credentials_dict_does_not_leak_either(self):
        client = UniversalAI(credentials={"api_key": "sk-deepseek"}, provider="deepseek")
        assert client._get_api_key(get_provider_config("deepseek")) == "sk-deepseek"
        assert client._get_api_key(get_provider_config("qwen")) is None

    def test_credentials_dict_is_copied(self, monkeypatch):
        # Mutating the caller's dict afterwards must not alter the client.
        supplied = {"api_key": "sk-original"}
        client = UniversalAI(credentials=supplied, provider="deepseek")
        supplied["api_key"] = "sk-mutated"
        assert client._get_api_key(get_provider_config("deepseek")) == "sk-original"

    def test_other_provider_uses_its_own_env_var(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-qwen-own")
        client = UniversalAI(api_key="sk-deepseek", provider="deepseek")

        assert client._get_api_key(get_provider_config("deepseek")) == "sk-deepseek"
        assert client._get_api_key(get_provider_config("qwen")) == "sk-qwen-own"

    def test_default_provider_falls_back_to_env_when_no_key_passed(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        client = UniversalAI(provider="deepseek")
        assert client._get_api_key(get_provider_config("deepseek")) == "sk-from-env"

    def test_rotated_env_key_is_picked_up_without_rebuilding(self, monkeypatch):
        # Keys are resolved per call rather than captured at construction,
        # so rotation does not require a new client.
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-old")
        client = UniversalAI(provider="deepseek")
        assert client._get_api_key(get_provider_config("deepseek")) == "sk-old"

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-rotated")
        assert client._get_api_key(get_provider_config("deepseek")) == "sk-rotated"

    def test_explicit_key_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        client = UniversalAI(api_key="sk-explicit", provider="deepseek")
        assert client._get_api_key(get_provider_config("deepseek")) == "sk-explicit"


class TestCrossProviderRequests:
    def test_cross_provider_call_sends_that_provider_key(self, monkeypatch):
        """A ``provider=`` override must authenticate with that provider's key."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-qwen-own")
        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["auth"] = headers["Authorization"]

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"id": "id-1", "choices": [{"message": {"content": "ok"}}], "usage": {}}

            return FakeResponse()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(api_key="sk-deepseek", provider="deepseek")
        client.chat(
            messages=[{"role": "user", "content": "Hi"}], provider="qwen", model="qwen-plus"
        )

        assert "dashscope" in captured["url"]
        assert captured["auth"] == "Bearer sk-qwen-own"
        assert "sk-deepseek" not in captured["auth"]

    def test_cross_provider_call_without_key_raises_before_network(self, monkeypatch):
        """No key for the target provider must fail loudly, not fall back."""

        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("a request was sent without a valid credential")

        monkeypatch.setattr(client_module.httpx, "post", explode)
        client = UniversalAI(api_key="sk-deepseek", provider="deepseek")

        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                provider="qwen",
                model="qwen-plus",
            )
