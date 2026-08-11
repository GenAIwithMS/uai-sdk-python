"""Unit tests for the config file loader (Sub-module 1.0.3).

Tests cover:
  - YAML and JSON file loading
  - UAI_CONFIG_PATH environment variable
  - Default file auto-discovery
  - Merge with hardcoded provider configs
  - Adding new providers from config
  - Caching behaviour
  - Graceful error handling on malformed files
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from uai.exceptions import ConfigError
from uai.registry import (
    PROVIDER_REGISTRY,
    apply_to_registry,
    clear_cache,
    find_config_file,
    get_config,
    get_provider_config,
    load_config,
    load_config_file,
)
from uai.registry.schema import AuthType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def yaml_override_config(tmp_path: Path) -> Path:
    """A YAML config that overrides DeepSeek's base_url and timeout."""
    config_content = """
providers:
  deepseek:
    base_url: "https://custom.deepseek.com/v1"
    timeout: 45.0
    max_retries: 5
"""
    path = tmp_path / "providers.yaml"
    path.write_text(config_content)
    return path


@pytest.fixture
def json_override_config(tmp_path: Path) -> Path:
    """A JSON config that overrides Qwen's default model."""
    config_content = {
        "providers": {
            "qwen": {
                "default_model": "qwen-turbo",
            }
        }
    }
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(config_content, indent=2))
    return path


@pytest.fixture
def new_provider_config(tmp_path: Path) -> Path:
    """A config file that adds a completely new provider."""
    config_content = """
providers:
  my-llm:
    display_name: "My Custom LLM"
    base_url: "https://api.my-llm.com/v1"
    auth_type: "bearer_token"
    api_key_env_var: "MY_LLM_KEY"
    default_model: "chat"
    models:
      chat:
        id: "chat"
        display_name: "Chat"
        context_window: 32000
        max_output_tokens: 8192
        pricing:
          input_cost_per_1k: 0.001
          output_cost_per_1k: 0.003
        capabilities:
          chat: true
          streaming: true
          tools: false
          vision: false
          embeddings: false
          audio: false
          reasoning: false
          rerank: false
          tts: false
          transcription: false
    timeout: 30.0
    max_retries: 3
"""
    path = tmp_path / "providers.yaml"
    path.write_text(config_content)
    return path


@pytest.fixture
def invalid_yaml_config(tmp_path: Path) -> Path:
    """A YAML file with a syntax error (tabs are not allowed in YAML)."""
    path = tmp_path / "providers.yaml"
    path.write_text("\tproviders:\n\t  deepseek:\n\t    base_url: 'https://test.com'\n")
    return path


@pytest.fixture
def invalid_json_config(tmp_path: Path) -> Path:
    """A JSON file with a syntax error."""
    path = tmp_path / "providers.json"
    path.write_text("{invalid json content}")
    return path


@pytest.fixture
def missing_providers_key(tmp_path: Path) -> Path:
    """A config file without the required 'providers' key."""
    path = tmp_path / "providers.yaml"
    path.write_text("settings:\n  debug: true\n")
    return path


@pytest.fixture
def clean_registry():
    """Snapshot and restore the global registry around tests."""
    from uai.registry import MVP_PROVIDERS, PROVIDER_ORDER

    saved_registry = PROVIDER_REGISTRY.copy()
    saved_order = PROVIDER_ORDER.copy()
    saved_mvp = MVP_PROVIDERS.copy()
    clear_cache()

    yield

    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(saved_registry)
    PROVIDER_ORDER.clear()
    PROVIDER_ORDER.extend(saved_order)
    MVP_PROVIDERS.clear()
    MVP_PROVIDERS.extend(saved_mvp)
    clear_cache()


# ---------------------------------------------------------------------------
# load_config_file — YAML
# ---------------------------------------------------------------------------


class TestLoadConfigYaml:
    def test_override_existing_provider(self, yaml_override_config):
        """Override base_url and timeout on a hardcoded provider."""
        clear_cache()
        configs = load_config_file(yaml_override_config)
        assert "deepseek" in configs
        ds = configs["deepseek"]
        assert ds.base_url == "https://custom.deepseek.com/v1"
        assert ds.timeout == 45.0
        assert ds.max_retries == 5
        # Untouched fields should match the hardcoded defaults
        assert ds.default_model == "deepseek-v4-flash"
        assert ds.auth_type == AuthType.BEARER_TOKEN

    def test_override_keeps_models(self, yaml_override_config):
        """Overriding a top-level field should not blow away the models dict."""
        clear_cache()
        configs = load_config_file(yaml_override_config)
        ds = configs["deepseek"]
        assert "deepseek-v4-flash" in ds.models
        assert "deepseek-v4-pro" in ds.models

    def test_override_preserves_aliases(self, yaml_override_config):
        clear_cache()
        configs = load_config_file(yaml_override_config)
        model = configs["deepseek"].models["deepseek-v4-flash"]
        assert "deepseek-chat-latest" in model.aliases  # retired id kept as alias


# ---------------------------------------------------------------------------
# load_config_file — JSON
# ---------------------------------------------------------------------------


class TestLoadConfigJson:
    def test_override_default_model(self, json_override_config):
        """Override Qwen's default_model via JSON."""
        clear_cache()
        configs = load_config_file(json_override_config)
        assert "qwen" in configs
        assert configs["qwen"].default_model == "qwen-turbo"
        # Untouched
        assert configs["qwen"].base_url.startswith("https://dashscope")

    def test_json_preserves_models(self, json_override_config):
        clear_cache()
        configs = load_config_file(json_override_config)
        assert "qwen-plus" in configs["qwen"].models
        assert "qwen-vl-max" in configs["qwen"].models


# ---------------------------------------------------------------------------
# load_config_file — new providers
# ---------------------------------------------------------------------------


class TestLoadConfigNewProvider:
    def test_add_new_provider(self, new_provider_config):
        clear_cache()
        configs = load_config_file(new_provider_config)
        assert "my-llm" in configs
        provider = configs["my-llm"]
        assert provider.name == "my-llm"
        assert provider.display_name == "My Custom LLM"
        assert provider.base_url == "https://api.my-llm.com/v1"
        assert provider.auth_type == AuthType.BEARER_TOKEN
        assert provider.api_key_env_var == "MY_LLM_KEY"
        assert provider.default_model == "chat"
        assert "chat" in provider.models

    def test_new_provider_model_capabilities(self, new_provider_config):
        clear_cache()
        configs = load_config_file(new_provider_config)
        model = configs["my-llm"].models["chat"]
        assert model.capabilities.chat is True
        assert model.capabilities.tools is False
        assert model.capabilities.vision is False


# ---------------------------------------------------------------------------
# apply_to_registry
# ---------------------------------------------------------------------------


class TestApplyToRegistry:
    def test_apply_override_to_global_registry(self, yaml_override_config, clean_registry):
        clear_cache()
        configs = load_config_file(yaml_override_config)
        applied = apply_to_registry(configs)
        assert applied == ["deepseek"]

        # The global registry should now reflect the override
        ds = PROVIDER_REGISTRY["deepseek"]
        assert ds.base_url == "https://custom.deepseek.com/v1"
        assert ds.timeout == 45.0

    def test_apply_new_provider_to_registry(self, new_provider_config, clean_registry):
        clear_cache()
        configs = load_config_file(new_provider_config)
        applied = apply_to_registry(configs)
        assert applied == ["my-llm"]

        assert "my-llm" in PROVIDER_REGISTRY
        assert "my-llm" in get_provider_config("my-llm").name

    def test_apply_does_not_remove_existing_providers(self, yaml_override_config, clean_registry):
        clear_cache()
        configs = load_config_file(yaml_override_config)
        apply_to_registry(configs)

        # All original providers should still be present
        assert "qwen" in PROVIDER_REGISTRY
        assert "glm" in PROVIDER_REGISTRY


# ---------------------------------------------------------------------------
# load_config (auto-discovery + caching)
# ---------------------------------------------------------------------------


class TestLoadConfigAutoDiscovery:
    def test_no_config_file_returns_empty(self, monkeypatch, clean_registry):
        """When no config file exists, return empty dict (use hardcoded)."""
        monkeypatch.delenv("UAI_CONFIG_PATH", raising=False)
        clear_cache()

        with patch("uai.registry.loader.find_config_file", return_value=None):
            configs = load_config()
        assert configs == {}

    def test_uses_config_path_env(self, yaml_override_config, monkeypatch, clean_registry):
        """UAI_CONFIG_PATH should take priority."""
        clear_cache()
        monkeypatch.setenv("UAI_CONFIG_PATH", str(yaml_override_config))
        configs = load_config()
        assert "deepseek" in configs
        assert configs["deepseek"].base_url == "https://custom.deepseek.com/v1"

    def test_explicit_path_overrides_env(self, yaml_override_config, monkeypatch, clean_registry):
        """Explicit path argument should override UAI_CONFIG_PATH."""
        clear_cache()
        monkeypatch.setenv("UAI_CONFIG_PATH", "/nonexistent/path.yaml")
        configs = load_config(path=yaml_override_config)
        assert "deepseek" in configs

    def test_caching_returns_same_object(self, yaml_override_config, clean_registry):
        clear_cache()
        first = load_config(path=yaml_override_config)
        second = load_config()  # should return cached
        assert first is second

    def test_clear_cache_forces_reload(self, yaml_override_config, clean_registry):
        clear_cache()
        first = load_config(path=yaml_override_config)
        clear_cache()
        second = load_config(path=yaml_override_config)
        assert first == second
        assert first is not second  # new object after re-read

    def test_get_config_returns_cached(self, yaml_override_config, clean_registry):
        clear_cache()
        loaded = load_config(path=yaml_override_config)
        got = get_config()
        assert got is loaded


# ---------------------------------------------------------------------------
# find_config_file
# ---------------------------------------------------------------------------


class TestFindConfigFile:
    def test_returns_none_when_no_file(self, monkeypatch, clean_registry):
        monkeypatch.delenv("UAI_CONFIG_PATH", raising=False)
        clear_cache()

        # Patch all possible locations to not exist

        def mock_expanduser(path_str):
            # Make home-relative paths not resolve
            return Path(str(path_str).replace("~", "/nonexistent_home"))

        with (
            patch.object(Path, "expanduser", mock_expanduser),
            patch.object(Path, "exists", lambda self: False),
        ):
            result = find_config_file()
        assert result is None

    def test_env_path_takes_priority(self, tmp_path, monkeypatch, clean_registry):
        config_file = tmp_path / "custom-config.yaml"
        config_file.write_text("providers: {}\n")
        monkeypatch.setenv("UAI_CONFIG_PATH", str(config_file))
        clear_cache()

        result = find_config_file()
        assert result is not None
        assert result.name == "custom-config.yaml"

    def test_env_path_missing_raises(self, monkeypatch, clean_registry):
        monkeypatch.setenv("UAI_CONFIG_PATH", "/nonexistent/file.yaml")
        clear_cache()
        with pytest.raises(ConfigError, match="file does not exist"):
            find_config_file()

    def test_env_path_unsupported_extension_raises(self, tmp_path, monkeypatch, clean_registry):
        config_file = tmp_path / "config.txt"
        config_file.write_text("test")
        monkeypatch.setenv("UAI_CONFIG_PATH", str(config_file))
        clear_cache()
        with pytest.raises(ConfigError, match="unsupported extension"):
            find_config_file()

    def test_finds_yaml_file(self, tmp_path, monkeypatch, clean_registry):
        config_file = tmp_path / "providers.yaml"
        config_file.write_text("providers: {}\n")
        monkeypatch.setenv("UAI_CONFIG_PATH", str(config_file))
        clear_cache()

        result = find_config_file()
        assert result == config_file


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestConfigErrors:
    def test_invalid_yaml_raises_config_error(self, invalid_yaml_config):
        clear_cache()
        with pytest.raises(ConfigError, match="YAML parsing error"):
            load_config_file(invalid_yaml_config)

    def test_invalid_json_raises_config_error(self, invalid_json_config):
        clear_cache()
        with pytest.raises(ConfigError, match="JSON parsing error"):
            load_config_file(invalid_json_config)

    def test_missing_providers_key_raises(self, missing_providers_key):
        clear_cache()
        with pytest.raises(ConfigError, match="top-level 'providers' key"):
            load_config_file(missing_providers_key)

    def test_missing_file_raises(self, tmp_path):
        clear_cache()
        nonexistent = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ConfigError, match="not found"):
            load_config_file(nonexistent)

    def test_unsupported_extension_raises(self, tmp_path):
        clear_cache()
        path = tmp_path / "config.toml"
        path.write_text("providers = {}")
        with pytest.raises(ConfigError, match="Unsupported configuration file format"):
            load_config_file(path)

    def test_provider_config_not_a_dict_raises(self, tmp_path):
        clear_cache()
        path = tmp_path / "providers.yaml"
        path.write_text('providers:\n  deepseek: "not a dict"\n')
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config_file(path)

    def test_invalid_provider_config_raises(self, tmp_path):
        """A new provider with missing required fields should fail validation."""
        clear_cache()
        path = tmp_path / "providers.yaml"
        path.write_text("""
providers:
  broken-provider:
    display_name: "Broken"
    # missing: base_url, auth_type, api_key_env_var, models, default_model
""")
        with pytest.raises(ConfigError, match="Validation failed"):
            load_config_file(path)

    def test_yaml_without_pyyaml_raises(self, tmp_path):
        """If PyYAML is missing, YAML files should raise ConfigError."""
        clear_cache()
        path = tmp_path / "providers.yaml"
        path.write_text("providers: {}\n")
        with (
            patch("uai.registry.loader._HAS_YAML", False),
            pytest.raises(ConfigError, match="PyYAML is required"),
        ):
            load_config_file(path)

    def test_unsupported_config_format(self, tmp_path):
        clear_cache()
        path = tmp_path / "config.ini"
        path.write_text("[section]\nkey=value")
        with pytest.raises(ConfigError, match="Unsupported configuration file format"):
            load_config_file(path)


# ---------------------------------------------------------------------------
# Merge edge cases
# ---------------------------------------------------------------------------


class TestMergeEdgeCases:
    def test_partial_override_keeps_hardcoded_models(self, tmp_path, clean_registry):
        """Overriding only base_url should keep all original models."""
        clear_cache()
        config = """
providers:
  deepseek:
    base_url: "https://override.example.com/v1"
"""
        path = tmp_path / "providers.yaml"
        path.write_text(config)
        configs = load_config_file(path)
        ds = configs["deepseek"]
        assert ds.base_url == "https://override.example.com/v1"
        assert "deepseek-v4-flash" in ds.models
        assert "deepseek-v4-pro" in ds.models
        assert ds.default_model == "deepseek-v4-flash"

    def test_override_regions(self, tmp_path, clean_registry):
        """New regional endpoint for an existing provider."""
        clear_cache()
        config = """
providers:
  qwen:
    regions:
      cn-shanghai:
        base_url: "https://custom-shanghai.example.com/v1"
"""
        path = tmp_path / "providers.yaml"
        path.write_text(config)
        configs = load_config_file(path)
        qwen = configs["qwen"]
        assert "cn-shanghai" in qwen.regions
        assert qwen.regions["cn-shanghai"].base_url == "https://custom-shanghai.example.com/v1"
        # Original regions preserved
        assert "cn-hangzhou" in qwen.regions

    def test_override_pricing_only(self, tmp_path, clean_registry):
        clear_cache()
        config = """
providers:
  deepseek:
    models:
      deepseek-v4-flash:
        id: "deepseek-v4-flash"
        display_name: "DeepSeek Chat"
        context_window: 128000
        max_output_tokens: 32000
        pricing:
          input_cost_per_1k: 0.010
          output_cost_per_1k: 0.020
        capabilities:
          chat: true
          streaming: true
          tools: true
          vision: false
          embeddings: true
          audio: false
          reasoning: false
          rerank: false
          tts: false
          transcription: false
        aliases: ["deepseek-v4-flash-latest"]
      deepseek-v4-pro:
        id: "deepseek-v4-pro"
        display_name: "DeepSeek Reasoner"
        context_window: 128000
        max_output_tokens: 32000
        pricing:
          input_cost_per_1k: 0.014
          output_cost_per_1k: 0.028
        capabilities:
          chat: true
          streaming: true
          tools: true
          vision: false
          embeddings: false
          audio: false
          reasoning: true
          rerank: false
          tts: false
          transcription: false
        aliases: ["deepseek-v4-pro-1"]
"""
        path = tmp_path / "providers.yaml"
        path.write_text(config)
        configs = load_config_file(path)
        ds = configs["deepseek"]
        assert ds.models["deepseek-v4-flash"].pricing.input_cost_per_1k == 0.010
        assert ds.models["deepseek-v4-flash"].pricing.output_cost_per_1k == 0.020
        # Reasoner pricing should come from the YAML too
        assert ds.models["deepseek-v4-pro"].pricing.input_cost_per_1k == 0.014

    def test_default_model_in_new_provider(self, tmp_path, clean_registry):
        """New provider must have a valid default_model."""
        clear_cache()
        config = """
providers:
  new-llm:
    display_name: "New LLM"
    base_url: "https://api.new-llm.com/v1"
    auth_type: "bearer_token"
    api_key_env_var: "NEW_LLM_KEY"
    default_model: "nonexistent-model"
    models:
      other-model:
        id: "other-model"
        display_name: "Other"
        context_window: 4096
        max_output_tokens: 2048
        capabilities:
          chat: true
"""
        path = tmp_path / "providers.yaml"
        path.write_text(config)

        # Pass-through is on by default, so an undeclared default is forwarded
        # to the provider rather than rejected — that is what lets a config
        # name a model this SDK has never heard of.  It warns, though.
        loaded = load_config_file(path)
        assert loaded["new-llm"].default_model == "nonexistent-model"

        strict = config.replace(
            'default_model: "nonexistent-model"',
            'default_model: "nonexistent-model"\n    allow_unknown_models: false',
        )
        strict_path = tmp_path / "providers-strict.yaml"
        strict_path.write_text(strict)
        with pytest.raises(ConfigError, match="default_model"):
            load_config_file(strict_path)


# ---------------------------------------------------------------------------
# ConfigError is a UAIError
# ---------------------------------------------------------------------------


class TestConfigErrorInheritance:
    def test_is_uai_error(self):
        from uai.exceptions import UAIError

        assert issubclass(ConfigError, UAIError)

    def test_message_preserved(self):
        err = ConfigError("something went wrong")
        assert str(err) == "something went wrong"
