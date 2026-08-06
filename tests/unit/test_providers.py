"""Unit tests for the provider registry (Sub-module 1.0.2 & 1.0.5).

These tests verify that every hardcoded provider config is valid,
that the registry lookup functions behave correctly, and that the
capability-gating utility works as expected.
"""

from __future__ import annotations

import pytest

from uai.exceptions import FeatureNotSupportedError
from uai.registry import (
    MVP_PROVIDERS,
    PROVIDER_REGISTRY,
    check_capability,
    get_default_model,
    get_model_info,
    get_provider_config,
    list_mvp_providers,
    list_providers,
    register_provider,
)
from uai.registry.schema import AuthType, ProviderCapabilities, ProviderConfig

# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------


class TestRegistryContents:
    """Validate the hardcoded provider registry at import time."""

    def test_all_expected_providers_exist(self):
        expected = {"deepseek", "qwen", "glm", "kimi", "stepfun", "doubao", "minimax", "hunyuan"}
        assert set(PROVIDER_REGISTRY.keys()) == expected

    def test_mvp_providers_are_subset(self):
        assert set(MVP_PROVIDERS) <= set(PROVIDER_REGISTRY.keys())
        assert set(MVP_PROVIDERS) == {"deepseek", "qwen"}

    @pytest.mark.parametrize(
        "provider_name",
        [
            "deepseek",
            "qwen",
            "glm",
            "kimi",
            "stepfun",
            "doubao",
            "minimax",
            "hunyuan",
        ],
    )
    def test_all_providers_use_bearer_token(self, provider_name):
        config = PROVIDER_REGISTRY[provider_name]
        assert config.auth_type == AuthType.BEARER_TOKEN

    @pytest.mark.parametrize(
        "provider_name",
        [
            "deepseek",
            "qwen",
            "glm",
            "kimi",
            "stepfun",
            "doubao",
            "minimax",
            "hunyuan",
        ],
    )
    def test_all_providers_have_valid_base_url(self, provider_name):
        config = PROVIDER_REGISTRY[provider_name]
        assert config.base_url.startswith("https://")

    @pytest.mark.parametrize(
        "provider_name",
        [
            "deepseek",
            "qwen",
            "glm",
            "kimi",
            "stepfun",
            "doubao",
            "minimax",
            "hunyuan",
        ],
    )
    def test_all_providers_have_models(self, provider_name):
        config = PROVIDER_REGISTRY[provider_name]
        assert len(config.models) > 0

    @pytest.mark.parametrize(
        "provider_name",
        [
            "deepseek",
            "qwen",
            "glm",
            "kimi",
            "stepfun",
            "doubao",
            "minimax",
            "hunyuan",
        ],
    )
    def test_default_model_exists(self, provider_name):
        config = PROVIDER_REGISTRY[provider_name]
        assert config.default_model in config.models

    @pytest.mark.parametrize(
        "provider_name",
        [
            "deepseek",
            "qwen",
            "glm",
            "kimi",
            "stepfun",
            "doubao",
            "minimax",
            "hunyuan",
        ],
    )
    def test_all_models_have_positive_context_window(self, provider_name):
        config = PROVIDER_REGISTRY[provider_name]
        for model in config.models.values():
            assert model.context_window > 0
            assert model.max_output_tokens > 0


# ---------------------------------------------------------------------------
# list_providers
# ---------------------------------------------------------------------------


class TestListProviders:
    def test_returns_list(self):
        result = list_providers()
        assert isinstance(result, list)
        assert len(result) == 8

    def test_all_mvp_in_list(self):
        providers = list_providers()
        assert "deepseek" in providers
        assert "qwen" in providers

    def test_all_known_providers_in_list(self):
        providers = list_providers()
        for name in ["deepseek", "qwen", "glm", "kimi", "stepfun", "doubao", "minimax", "hunyuan"]:
            assert name in providers


# ---------------------------------------------------------------------------
# list_mvp_providers
# ---------------------------------------------------------------------------


class TestListMvpProviders:
    def test_returns_only_mvp(self):
        result = list_mvp_providers()
        assert set(result) == {"deepseek", "qwen"}


# ---------------------------------------------------------------------------
# get_provider_config
# ---------------------------------------------------------------------------


class TestGetProviderConfig:
    def test_get_deepseek(self):
        config = get_provider_config("deepseek")
        assert config.name == "deepseek"
        assert config.display_name == "DeepSeek AI"
        assert config.base_url == "https://api.deepseek.com/v1"
        assert config.api_key_env_var == "DEEPSEEK_API_KEY"

    def test_get_qwen(self):
        config = get_provider_config("qwen")
        assert config.name == "qwen"
        assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_case_insensitive_lookup(self):
        config = get_provider_config("DEEPSEEK")
        assert config.name == "deepseek"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            get_provider_config("nonexistent")

    def test_unknown_provider_lists_available(self):
        with pytest.raises(ValueError, match="deepseek, qwen"):
            get_provider_config("invalid-provider")


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    def test_get_deepseek_chat_model(self):
        model = get_model_info("deepseek", "deepseek-chat")
        assert model.id == "deepseek-chat"
        assert model.capabilities.chat is True
        assert model.capabilities.vision is False

    def test_get_deepseek_by_alias(self):
        model = get_model_info("deepseek", "deepseek-chat-latest")
        assert model.id == "deepseek-chat"

    def test_get_qwen_vl_model(self):
        model = get_model_info("qwen", "qwen-vl-max")
        assert model.capabilities.vision is True
        assert model.capabilities.embeddings is False

    def test_get_embedding_model(self):
        model = get_model_info("qwen", "text-embedding-v4")
        assert model.capabilities.embeddings is True
        assert model.capabilities.chat is False

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            get_model_info("nonexistent", "some-model")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_model_info("deepseek", "nonexistent-model")


# ---------------------------------------------------------------------------
# get_default_model
# ---------------------------------------------------------------------------


class TestGetDefaultModel:
    def test_deepseek_default(self):
        assert get_default_model("deepseek") == "deepseek-chat"

    def test_qwen_default(self):
        assert get_default_model("qwen") == "qwen-plus"

    def test_glm_default(self):
        assert get_default_model("glm") == "glm-4.7"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            get_default_model("nonexistent")


# ---------------------------------------------------------------------------
# check_capability
# ---------------------------------------------------------------------------


class TestCheckCapability:
    def test_supported_capability_passes(self):
        # Should not raise
        check_capability("qwen", "qwen-vl-max", "vision")

    def test_unsupported_capability_raises(self):
        with pytest.raises(FeatureNotSupportedError, match="vision"):
            check_capability("deepseek", "deepseek-chat", "vision")

    def test_unsupported_capability_on_correct_provider(self):
        # deepseek-chat supports embeddings; vision is NOT supported
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            check_capability("deepseek", "deepseek-chat", "vision")
        assert "vision" in str(exc_info.value)
        assert "deepseek" in str(exc_info.value)

    def test_reasoner_capability(self):
        # deepseek-reasoner supports reasoning, deepseek-chat does not
        check_capability("deepseek", "deepseek-reasoner", "reasoning")
        with pytest.raises(FeatureNotSupportedError, match="reasoning"):
            check_capability("deepseek", "deepseek-chat", "reasoning")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            check_capability("nonexistent", "model", "chat")

    def test_unknown_capability_name(self):
        with pytest.raises(ValueError, match="Unknown capability"):
            check_capability("deepseek", "deepseek-chat", "time_travel")

    def test_minimax_audio_capabilities(self):
        # MiniMax supports TTS and transcription
        check_capability("minimax", "minimax-m2.5", "tts")
        check_capability("minimax", "minimax-m2.5", "transcription")

    def test_glm_no_streaming_for_embedding(self):
        with pytest.raises(FeatureNotSupportedError, match="streaming"):
            check_capability("glm", "glm-embedding", "streaming")


# ---------------------------------------------------------------------------
# register_provider
# ---------------------------------------------------------------------------


class TestRegisterProvider:
    def test_register_new_provider(self):
        from uai.registry.schema import ProviderModel

        custom = ProviderConfig(
            name="custom-test",
            display_name="Custom Test",
            base_url="https://api.custom.test/v1",
            auth_type=AuthType.BEARER_TOKEN,
            api_key_env_var="CUSTOM_TEST_KEY",
            models={
                "test-model": ProviderModel(
                    id="test-model",
                    display_name="Test Model",
                    context_window=4096,
                    max_output_tokens=2048,
                    capabilities=ProviderCapabilities(chat=True, streaming=True),
                )
            },
            default_model="test-model",
        )
        result = register_provider(custom)
        assert result.name == "custom-test"
        assert "custom-test" in list_providers()

        # Cleanup
        PROVIDER_REGISTRY.pop("custom-test", None)
        from uai.registry.providers import PROVIDER_ORDER

        if "custom-test" in PROVIDER_ORDER:
            PROVIDER_ORDER.remove("custom-test")

    def test_register_existing_provider_without_override_raises(self):
        config = get_provider_config("deepseek")
        with pytest.raises(ValueError, match="already registered"):
            register_provider(config, override=False)

    def test_register_existing_with_override(self):
        original = get_provider_config("deepseek")
        modified = original.model_copy(update={"default_model": "deepseek-reasoner"})
        result = register_provider(modified, override=True)
        assert result.default_model == "deepseek-reasoner"

        # Restore original
        register_provider(original, override=True)


# ---------------------------------------------------------------------------
# Capability matrix summary tests
# ---------------------------------------------------------------------------


class TestCapabilityMatrix:
    """Verify the capability matrix matches the SRS / Implementation Plan."""

    @pytest.mark.parametrize(
        "provider_name,capabilities",
        [
            ("deepseek", {"chat", "streaming", "tools", "reasoning"}),
            ("qwen", {"chat", "streaming", "tools", "vision", "embeddings", "rerank"}),
            ("glm", {"chat", "streaming", "tools", "embeddings", "rerank"}),
            ("kimi", {"chat", "streaming", "tools"}),
            ("stepfun", {"chat", "streaming", "tools", "vision", "embeddings"}),
            ("doubao", {"chat", "streaming", "tools", "vision", "embeddings"}),
            ("minimax", {"chat", "streaming", "tools", "embeddings", "tts", "transcription"}),
            ("hunyuan", {"chat", "streaming", "tools", "vision", "embeddings"}),
        ],
    )
    def test_aggregate_capabilities(self, provider_name, capabilities):
        config = get_provider_config(provider_name)
        agg = config.capabilities
        for cap in capabilities:
            assert getattr(agg, cap) is True, f"{provider_name} should support {cap}"

    @pytest.mark.parametrize(
        "provider_name,disabled_cap",
        [
            ("deepseek", "vision"),
            ("glm", "vision"),
            ("kimi", "vision"),
            ("doubao", "tts"),
            ("hunyuan", "tts"),
        ],
    )
    def test_aggregate_capabilities_disabled(self, provider_name, disabled_cap):
        config = get_provider_config(provider_name)
        agg = config.capabilities
        assert getattr(agg, disabled_cap) is False
