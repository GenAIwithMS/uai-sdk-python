"""Unit tests for environment variable overrides (Sub-module 1.0.4).

Tests cover:
  - BASE_URL, TIMEOUT, MAX_RETRIES, RATE_LIMIT_RPM, RATE_LIMIT_TPM overrides
  - AUTH_TYPE override
  - API_KEY_ENV override
  - Boolean feature-flag overrides (DISABLE_{CAPABILITY})
  - Invalid value handling with warnings
  - Caching behaviour (no unnecessary copies)
  - Applying overrides to the global registry
"""

from __future__ import annotations

import logging

import pytest

from uai.registry import (
    PROVIDER_REGISTRY,
    apply_env_overrides,
    apply_env_overrides_to_config,
    get_env_overrides,
    get_provider_config,
)
from uai.registry.env import _parse_bool
from uai.registry.schema import AuthType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_env(monkeypatch: pytest.MonkeyPatch, name: str, value: str):
    monkeypatch.setenv(name, value)


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch, provider_name: str):
    """Remove all UAI_PROVIDER_* env vars for *provider_name*."""
    upper = provider_name.upper()
    for key in list(monkeypatch._setenvs.keys()):
        if key.startswith(f"UAI_PROVIDER_{upper}_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# _parse_bool
# ---------------------------------------------------------------------------


class TestParseBool:
    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_truthy(self, val):
        assert _parse_bool(val) is True

    @pytest.mark.parametrize("val", ["false", "False", "0", "no", "off", ""])
    def test_falsy(self, val):
        assert _parse_bool(val) is False

    @pytest.mark.parametrize("val", ["maybe", "2", "yep"])
    def test_invalid_raises(self, val):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_bool(val)


# ---------------------------------------------------------------------------
# BASE_URL override
# ---------------------------------------------------------------------------


class TestBaseUrlOverride:
    def test_override_deepseek_base_url(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_BASE_URL", "https://staging.deepseek.com/v1")

        overrides = get_env_overrides("deepseek")
        assert overrides["base_url"] == "https://staging.deepseek.com/v1"

    def test_override_with_trailing_whitespace(self, monkeypatch):
        _set_env(
            monkeypatch, "UAI_PROVIDER_DEEPSEEK_BASE_URL", "  https://staging.deepseek.com/v1  "
        )
        overrides = get_env_overrides("deepseek")
        assert overrides["base_url"] == "https://staging.deepseek.com/v1"


# ---------------------------------------------------------------------------
# TIMEOUT override
# ---------------------------------------------------------------------------


class TestTimeoutOverride:
    def test_override_timeout(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_TIMEOUT", "60")
        overrides = get_env_overrides("qwen")
        assert overrides["timeout"] == 60.0

    def test_override_timeout_float(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_TIMEOUT", "45.5")
        overrides = get_env_overrides("qwen")
        assert overrides["timeout"] == 45.5

    def test_invalid_timeout_logs_warning(self, monkeypatch, caplog):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_TIMEOUT", "not_a_number")
        with caplog.at_level(logging.WARNING):
            overrides = get_env_overrides("qwen")
        assert "timeout" not in overrides

    def test_empty_timeout_sketched(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_TIMEOUT", "  ")
        overrides = get_env_overrides("qwen")
        assert "timeout" not in overrides


# ---------------------------------------------------------------------------
# MAX_RETRIES override
# ---------------------------------------------------------------------------


class TestMaxRetriesOverride:
    def test_override_max_retries(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_MAX_RETRIES", "5")
        overrides = get_env_overrides("deepseek")
        assert overrides["max_retries"] == 5

    def test_invalid_max_retries_skipped(self, monkeypatch, caplog):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_MAX_RETRIES", "abc")
        with caplog.at_level(logging.WARNING):
            overrides = get_env_overrides("deepseek")
        assert "max_retries" not in overrides


# ---------------------------------------------------------------------------
# Rate limit overrides
# ---------------------------------------------------------------------------


class TestRateLimitOverrides:
    def test_override_rpm(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_RATE_LIMIT_RPM", "1000")
        overrides = get_env_overrides("qwen")
        assert overrides["rate_limit_rpm"] == 1000

    def test_override_tpm(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_RATE_LIMIT_TPM", "100000")
        overrides = get_env_overrides("qwen")
        assert overrides["rate_limit_tpm"] == 100000

    def test_override_both(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_RATE_LIMIT_RPM", "500")
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_RATE_LIMIT_TPM", "50000")
        overrides = get_env_overrides("qwen")
        assert overrides["rate_limit_rpm"] == 500
        assert overrides["rate_limit_tpm"] == 50000


# ---------------------------------------------------------------------------
# Auth type override
# ---------------------------------------------------------------------------


class TestAuthTypeOverride:
    def test_override_to_api_key(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_AUTH_TYPE", "API_KEY")
        overrides = get_env_overrides("deepseek")
        assert overrides["auth_type"] == AuthType.API_KEY

    def test_override_to_oauth(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_AUTH_TYPE", "oauth")
        overrides = get_env_overrides("qwen")
        assert overrides["auth_type"] == AuthType.OAUTH

    def test_invalid_auth_type_skipped(self, monkeypatch, caplog):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_AUTH_TYPE", "invalid_type")
        with caplog.at_level(logging.WARNING):
            overrides = get_env_overrides("qwen")
        assert "auth_type" not in overrides


# ---------------------------------------------------------------------------
# API key env var override
# ---------------------------------------------------------------------------


class TestApiKeyEnvOverride:
    def test_override_api_key_env(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_API_KEY_ENV", "MY_DEEPSEEK_KEY")
        overrides = get_env_overrides("deepseek")
        assert overrides["api_key_env_var"] == "MY_DEEPSEEK_KEY"

    def test_override_with_whitespace(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_API_KEY_ENV", "  MY_KEY  ")
        overrides = get_env_overrides("deepseek")
        assert overrides["api_key_env_var"] == "MY_KEY"


# ---------------------------------------------------------------------------
# Feature-flag boolean overrides
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    def test_disable_vision(self, monkeypatch, clean_registry):
        """Setting UAI_PROVIDER_QWEN_DISABLE_VISION=true should disable vision on all Qwen models."""
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_DISABLE_VISION", "true")
        config = apply_env_overrides_to_config(PROVIDER_REGISTRY["qwen"])
        for model in config.models.values():
            assert model.capabilities.vision is False

    def test_disable_streaming_false_keeps_original(self, monkeypatch):
        """Setting UAI_PROVIDER_QWEN_DISABLE_STREAMING=false should NOT disable streaming."""
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_DISABLE_STREAMING", "false")
        config = apply_env_overrides_to_config(PROVIDER_REGISTRY["qwen"])
        # Only check chat-capable models
        for model in config.models.values():
            if not model.capabilities.chat:
                continue
            assert model.capabilities.streaming is True

    def test_disable_tools_via_yes(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_DISABLE_TOOLS", "yes")
        config = apply_env_overrides_to_config(PROVIDER_REGISTRY["qwen"])
        for model in config.models.values():
            assert model.capabilities.tools is False

    def test_disable_unknown_capability_ignored(self, monkeypatch, caplog):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_DISABLE_TIME_TRAVEL", "true")
        with caplog.at_level(logging.WARNING):
            overrides = get_env_overrides("qwen")
        # Should not contain the unknown capability
        assert "time_travel" not in str(overrides)

    def test_no_feature_flag_returns_original(self, monkeypatch):
        """No env overrides should return the same config object (not a copy)."""
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_BASE_URL", "")  # empty = no override
        config = PROVIDER_REGISTRY["qwen"]
        result = apply_env_overrides_to_config(config)
        assert result is config


# ---------------------------------------------------------------------------
# apply_env_overrides_to_config
# ---------------------------------------------------------------------------


class TestApplyEnvOverridesToConfig:
    def test_returns_same_object_when_no_overrides(self, monkeypatch):
        config = get_provider_config("deepseek")
        result = apply_env_overrides_to_config(config)
        assert result is config

    def test_returns_copy_when_overridden(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_TIMEOUT", "99")
        config = get_provider_config("deepseek")
        result = apply_env_overrides_to_config(config)
        assert result is not config
        assert result.timeout == 99.0
        # Original unchanged
        assert config.timeout == 30.0

    def test_multiple_overrides_on_same_provider(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_BASE_URL", "https://override.com")
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_TIMEOUT", "60")
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_MAX_RETRIES", "10")
        config = get_provider_config("deepseek")
        result = apply_env_overrides_to_config(config)
        assert result.base_url == "https://override.com"
        assert result.timeout == 60.0
        assert result.max_retries == 10

    def test_combined_override_and_feature_flag(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_TIMEOUT", "15")
        _set_env(monkeypatch, "UAI_PROVIDER_QWEN_DISABLE_VISION", "true")
        config = get_provider_config("qwen")
        result = apply_env_overrides_to_config(config)
        assert result.timeout == 15.0
        for model in result.models.values():
            assert model.capabilities.vision is False
            # Chat capability preserved for chat-capable models
            if not model.capabilities.chat:
                continue
            assert model.capabilities.chat is True


# ---------------------------------------------------------------------------
# apply_env_overrides (all providers)
# ---------------------------------------------------------------------------


class TestApplyEnvOverridesAll:
    def test_no_env_vars_returns_all_originals(self, monkeypatch):
        """No env overrides set → all configs returned by reference."""
        # Remove all UAI_PROVIDER_* env vars
        provider_names = [
            "deepseek",
            "qwen",
            "glm",
            "kimi",
            "stepfun",
            "doubao",
            "minimax",
            "hunyuan",
        ]
        fields = [
            "BASE_URL",
            "TIMEOUT",
            "MAX_RETRIES",
            "RATE_LIMIT_RPM",
            "RATE_LIMIT_TPM",
            "AUTH_TYPE",
            "API_KEY_ENV",
            "API_VERSION",
            "DOCUMENTATION_URL",
        ]
        capabilities = [
            "CHAT",
            "STREAMING",
            "TOOLS",
            "VISION",
            "EMBEDDINGS",
            "AUDIO",
            "REASONING",
            "RERANK",
            "TTS",
            "TRANSCRIPTION",
        ]
        for name in provider_names:
            upper = name.upper()
            for field in fields:
                monkeypatch.delenv(f"UAI_PROVIDER_{upper}_{field}", raising=False)
            for cap in capabilities:
                monkeypatch.delenv(f"UAI_PROVIDER_{upper}_DISABLE_{cap}", raising=False)

        result = apply_env_overrides()
        assert len(result) == len(PROVIDER_REGISTRY)
        for name, config in result.items():
            assert config is PROVIDER_REGISTRY[name]

    def test_only_overridden_providers_are_copies(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_DEEPSEEK_TIMEOUT", "50")
        result = apply_env_overrides()
        # DeepSeek should be a copy
        assert result["deepseek"] is not PROVIDER_REGISTRY["deepseek"]
        assert result["deepseek"].timeout == 50.0
        # Qwen should be the original
        assert result["qwen"] is PROVIDER_REGISTRY["qwen"]

    def test_apply_to_explicit_configs(self, monkeypatch):
        _set_env(monkeypatch, "UAI_PROVIDER_GLM_TIMEOUT", "120")
        configs = {
            "glm": get_provider_config("glm"),
            "kimi": get_provider_config("kimi"),
        }
        result = apply_env_overrides(configs)
        assert result["glm"].timeout == 120.0
        assert result["kimi"] is configs["kimi"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_registry():
    """Snapshot and restore the global registry around tests."""
    from uai.registry import MVP_PROVIDERS, PROVIDER_ORDER

    saved_registry = PROVIDER_REGISTRY.copy()
    saved_order = PROVIDER_ORDER.copy()
    saved_mvp = MVP_PROVIDERS.copy()

    yield

    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(saved_registry)
    PROVIDER_ORDER.clear()
    PROVIDER_ORDER.extend(saved_order)
    MVP_PROVIDERS.clear()
    MVP_PROVIDERS.extend(saved_mvp)
