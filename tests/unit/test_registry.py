"""Unit tests for the provider registry schema (Sub-module 1.0.1 & 1.0.5).

These tests validate every rule described in the Implementation Plan's
"Sub-module 1.0.5: Registry Validation and Testing" section.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uai.registry.schema import (
    AuthType,
    ProviderCapabilities,
    ProviderConfig,
    ProviderModel,
    ProviderPricing,
    RegionConfig,
)
from uai.exceptions import FeatureNotSupportedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    )


@pytest.fixture
def sample_pricing() -> ProviderPricing:
    return ProviderPricing(input_cost_per_1k=0.14, output_cost_per_1k=0.28)


@pytest.fixture
def sample_model(sample_capabilities, sample_pricing) -> ProviderModel:
    return ProviderModel(
        id="deepseek-chat",
        display_name="DeepSeek Chat",
        context_window=128_000,
        max_output_tokens=32_000,
        pricing=sample_pricing,
        capabilities=sample_capabilities,
        aliases=["deepseek-chat-1", "deepseek-chat-latest"],
    )


@pytest.fixture
def sample_config(sample_model) -> ProviderConfig:
    return ProviderConfig(
        name="deepseek",
        display_name="DeepSeek AI",
        base_url="https://api.deepseek.com/v1",
        auth_type=AuthType.BEARER_TOKEN,
        api_key_env_var="DEEPSEEK_API_KEY",
        models={"deepseek-chat": sample_model},
        default_model="deepseek-chat",
    )


# ---------------------------------------------------------------------------
# AuthType enum
# ---------------------------------------------------------------------------

class TestAuthType:
    def test_all_members(self):
        assert {e.name for e in AuthType} == {"API_KEY", "BEARER_TOKEN", "OAUTH"}

    def test_values_are_strings(self):
        for member in AuthType:
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# ProviderCapabilities
# ---------------------------------------------------------------------------

class TestProviderCapabilities:
    def test_defaults_all_false(self):
        caps = ProviderCapabilities()
        for field_name in ProviderCapabilities.model_fields:
            assert getattr(caps, field_name) is False

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ProviderCapabilities(chat=True, bogus=True)

    def test_all_fields_boolean(self):
        caps = ProviderCapabilities(
            chat=True,
            streaming=True,
            tools=True,
            vision=True,
            embeddings=True,
            audio=True,
            reasoning=True,
            rerank=True,
            tts=True,
            transcription=True,
        )
        assert caps.chat is True
        assert caps.tts is True


# ---------------------------------------------------------------------------
# ProviderPricing
# ---------------------------------------------------------------------------

class TestProviderPricing:
    def test_defaults_to_zero(self):
        pricing = ProviderPricing()
        assert pricing.input_cost_per_1k == 0.0
        assert pricing.output_cost_per_1k == 0.0

    def test_negative_input_rejected(self):
        with pytest.raises(ValidationError):
            ProviderPricing(input_cost_per_1k=-0.1)

    def test_negative_output_rejected(self):
        with pytest.raises(ValidationError):
            ProviderPricing(output_cost_per_1k=-0.5)

    def test_cost_calculation(self, sample_pricing):
        cost = sample_pricing.cost_for(input_tokens=500, output_tokens=500)
        # 500 / 1000 * 0.14 + 500 / 1000 * 0.28 = 0.07 + 0.14 = 0.21
        assert cost == pytest.approx(0.21)

    def test_cost_zero_tokens(self, sample_pricing):
        assert sample_pricing.cost_for(0, 0) == 0.0


# ---------------------------------------------------------------------------
# ProviderModel
# ---------------------------------------------------------------------------

class TestProviderModel:
    def test_valid_model(self, sample_model):
        assert sample_model.id == "deepseek-chat"
        assert sample_model.display_name == "DeepSeek Chat"
        assert sample_model.context_window == 128_000
        assert sample_model.max_output_tokens == 32_000
        assert len(sample_model.aliases) == 2

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            ProviderModel(
                id="  ",
                display_name="Test",
                context_window=1000,
                max_output_tokens=500,
            )

    def test_empty_display_name_rejected(self):
        with pytest.raises(ValidationError):
            ProviderModel(
                id="test",
                display_name="",
                context_window=1000,
                max_output_tokens=500,
            )

    def test_context_window_must_be_positive(self):
        with pytest.raises(ValidationError):
            ProviderModel(
                id="m", display_name="M", context_window=0, max_output_tokens=500
            )

    def test_max_output_must_be_positive(self):
        with pytest.raises(ValidationError):
            ProviderModel(
                id="m", display_name="M", context_window=1000, max_output_tokens=-1
            )

    def test_id_stripped(self):
        model = ProviderModel(
            id="  deepseek-chat  ",
            display_name="  Test  ",
            context_window=1000,
            max_output_tokens=500,
        )
        assert model.id == "deepseek-chat"
        assert model.display_name == "Test"


# ---------------------------------------------------------------------------
# RegionConfig
# ---------------------------------------------------------------------------

class TestRegionConfig:
    def test_valid_region(self):
        region = RegionConfig(base_url="https://api.example.com/v1")
        assert region.base_url == "https://api.example.com/v1"
        assert region.auth_type is None  # defaults to provider-level

    def test_region_with_auth_override(self):
        region = RegionConfig(
            base_url="https://cn.example.com/v1",
            auth_type=AuthType.OAUTH,
            api_key_env_var="CN_API_KEY",
        )
        assert region.auth_type == AuthType.OAUTH
        assert region.api_key_env_var == "CN_API_KEY"

    def test_empty_base_url_rejected(self):
        with pytest.raises(ValidationError):
            RegionConfig(base_url="  ")


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

class TestProviderConfig:
    def test_valid_config(self, sample_config):
        assert sample_config.name == "deepseek"
        assert sample_config.display_name == "DeepSeek AI"
        assert sample_config.base_url == "https://api.deepseek.com/v1"
        assert sample_config.auth_type == AuthType.BEARER_TOKEN
        assert sample_config.api_key_env_var == "DEEPSEEK_API_KEY"
        assert sample_config.default_model == "deepseek-chat"
        assert sample_config.api_version == "v1"
        assert sample_config.timeout == 30.0
        assert sample_config.max_retries == 3

    # -- name validation ---------------------------------------------------

    def test_name_lowercased(self):
        config = ProviderConfig(
            name="DEEPSEEK",
            display_name="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            auth_type=AuthType.BEARER_TOKEN,
            api_key_env_var="DEEPSEEK_API_KEY",
            models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
            default_model="m",
        )
        assert config.name == "deepseek"

    def test_name_with_spaces_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="deep seek",
                display_name="DeepSeek",
                base_url="https://api.deepseek.com/v1",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="DEEPSEEK_API_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
            )

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="  ",
                display_name="DeepSeek",
                base_url="https://api.deepseek.com/v1",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="DEEPSEEK_API_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
            )

    # -- base_url validation -----------------------------------------------

    def test_invalid_base_url_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                display_name="Test",
                base_url="ftp://api.deepseek.com",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="TEST_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
            )

    def test_empty_base_url_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                display_name="Test",
                base_url="  ",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="TEST_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
            )

    # -- default_model validation ------------------------------------------

    def test_default_model_not_in_models_rejected(self):
        with pytest.raises(ValidationError, match="default_model"):
            ProviderConfig(
                name="test",
                display_name="Test",
                base_url="https://api.test.com/v1",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="TEST_KEY",
                models={"a": ProviderModel(id="a", display_name="A", context_window=10, max_output_tokens=5)},
                default_model="nonexistent",
            )

    # -- timeout / retries -------------------------------------------------

    @pytest.mark.parametrize("timeout", [0, -1, 301])
    def test_invalid_timeout_rejected(self, timeout):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                display_name="Test",
                base_url="https://api.test.com/v1",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="TEST_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
                timeout=timeout,
            )

    @pytest.mark.parametrize("retries", [-1, 11])
    def test_invalid_max_retries_rejected(self, retries):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                display_name="Test",
                base_url="https://api.test.com/v1",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="TEST_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
                max_retries=retries,
            )

    # -- rate limits -------------------------------------------------------

    def test_negative_rate_limit_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                display_name="Test",
                base_url="https://api.test.com/v1",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="TEST_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
                rate_limit_rpm=-10,
            )

    # -- extra fields ------------------------------------------------------

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                display_name="Test",
                base_url="https://api.test.com/v1",
                auth_type=AuthType.BEARER_TOKEN,
                api_key_env_var="TEST_KEY",
                models={"m": ProviderModel(id="m", display_name="M", context_window=10, max_output_tokens=5)},
                default_model="m",
                rogue_field=True,
            )

    # -- capabilities aggregation ------------------------------------------

    def test_capabilities_aggregate(self, sample_model):
        model2 = ProviderModel(
            id="deepseek-reasoner",
            display_name="DeepSeek Reasoner",
            context_window=128_000,
            max_output_tokens=32_000,
            capabilities=ProviderCapabilities(chat=True, streaming=True, tools=True, reasoning=True),
            aliases=[],
        )
        config = ProviderConfig(
            name="deepseek",
            display_name="DeepSeek AI",
            base_url="https://api.deepseek.com/v1",
            auth_type=AuthType.BEARER_TOKEN,
            api_key_env_var="DEEPSEEK_API_KEY",
            models={
                "deepseek-chat": sample_model,
                "deepseek-reasoner": model2,
            },
            default_model="deepseek-chat",
        )
        caps = config.capabilities
        assert caps.chat is True
        assert caps.embeddings is True  # only in sample_model
        assert caps.reasoning is True  # only in model2
        assert caps.vision is False

    # -- get_model ---------------------------------------------------------

    def test_get_model_by_id(self, sample_config):
        model = sample_config.get_model("deepseek-chat")
        assert model.id == "deepseek-chat"

    def test_get_model_by_alias(self, sample_config):
        model = sample_config.get_model("deepseek-chat-latest")
        assert model.id == "deepseek-chat"

    def test_get_model_not_found(self, sample_config):
        with pytest.raises(ValueError, match="not found"):
            sample_config.get_model("nonexistent")

    # -- all_model_ids -----------------------------------------------------

    def test_all_model_ids_includes_aliases(self, sample_config):
        ids = sample_config.all_model_ids
        assert "deepseek-chat" in ids
        assert "deepseek-chat-1" in ids
        assert "deepseek-chat-latest" in ids


# ---------------------------------------------------------------------------
# Integration: capability enforcement
# ---------------------------------------------------------------------------

class TestCapabilityEnforcement:
    """Verify that the schema supports capability-aware request gating."""

    def test_vision_not_supported(self):
        caps = ProviderCapabilities(chat=True, streaming=True)
        assert caps.vision is False

    def test_feature_not_supported_error(self):
        error = FeatureNotSupportedError(
            feature="vision",
            provider="deepseek",
            model="deepseek-chat",
        )
        assert "vision" in str(error)
        assert "deepseek" in str(error)
        assert "deepseek-chat" in str(error)


# ---------------------------------------------------------------------------
# Multiple providers with different capability sets
# ---------------------------------------------------------------------------

class TestMultiProviderCapabilities:
    @pytest.fixture
    def qwen_config(self) -> ProviderConfig:
        return ProviderConfig(
            name="qwen",
            display_name="Qwen",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            auth_type=AuthType.BEARER_TOKEN,
            api_key_env_var="DASHSCOPE_API_KEY",
            models={
                "qwen-plus": ProviderModel(
                    id="qwen-plus",
                    display_name="Qwen Plus",
                    context_window=128_000,
                    max_output_tokens=8_192,
                    pricing=ProviderPricing(input_cost_per_1k=0.002, output_cost_per_1k=0.006),
                    capabilities=ProviderCapabilities(
                        chat=True, streaming=True, tools=True,
                        vision=True, embeddings=True, rerank=True,
                    ),
                    aliases=["qwen-plus-0824"],
                )
            },
            default_model="qwen-plus",
        )

    def test_qwen_vision_enabled(self, qwen_config):
        assert qwen_config.capabilities.vision is True

    def test_qwen_audio_disabled(self, qwen_config):
        assert qwen_config.capabilities.audio is False

    def test_deepseek_vision_disabled(self, sample_config):
        assert sample_config.capabilities.vision is False

    def test_qwen_cost_calculation(self, qwen_config):
        model = qwen_config.get_model("qwen-plus")
        cost = model.pricing.cost_for(input_tokens=10_000, output_tokens=1_000)
        # 10 * 0.002 + 1 * 0.006 = 0.026
        assert cost == pytest.approx(0.026)
