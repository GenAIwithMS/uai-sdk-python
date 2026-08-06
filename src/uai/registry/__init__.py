"""Provider registry sub-package for the Universal AI Provider SDK.

Public API:
    - ``PROVIDER_REGISTRY`` — dict of all registered ProviderConfig objects
    - ``MVP_PROVIDERS`` — providers targeted for the MVP release
    - ``get_provider_config(name)`` — look up a provider by name
    - ``list_providers()`` — list all registered provider names
    - ``list_mvp_providers()`` — list MVP-only provider names
    - ``get_model_info(provider, model_id)`` — look up a model's metadata
    - ``get_default_model(provider)`` — get the default model for a provider
    - ``register_provider(config)`` — register a custom provider at runtime
    - ``check_capability(provider, model, capability)`` — assert support
"""

from .schema import (
    AuthType,
    ProviderCapabilities,
    ProviderConfig,
    ProviderModel,
    ProviderPricing,
    RegionConfig,
)
from .providers import (
    PROVIDER_REGISTRY,
    MVP_PROVIDERS,
    PROVIDER_ORDER,
    DEEPSEEK_CONFIG,
    QWEN_CONFIG,
    GLM_CONFIG,
    KIMI_CONFIG,
    STEPFUN_CONFIG,
    DOUBAO_CONFIG,
    MINIMAX_CONFIG,
    HUNYUAN_CONFIG,
    get_provider_config,
    list_providers,
    list_mvp_providers,
    get_model_info,
    get_default_model,
    register_provider,
    check_capability,
)
from .loader import (
    find_config_file,
    load_config_file,
    load_config,
    get_config,
    clear_cache,
    apply_to_registry,
    ConfigError,
)
from .env import (
    get_env_overrides,
    apply_env_overrides_to_config,
    apply_env_overrides,
)

__all__ = [
    "AuthType",
    "ProviderCapabilities",
    "ProviderConfig",
    "ProviderModel",
    "ProviderPricing",
    "RegionConfig",
    "PROVIDER_REGISTRY",
    "MVP_PROVIDERS",
    "PROVIDER_ORDER",
    "DEEPSEEK_CONFIG",
    "QWEN_CONFIG",
    "GLM_CONFIG",
    "KIMI_CONFIG",
    "STEPFUN_CONFIG",
    "DOUBAO_CONFIG",
    "MINIMAX_CONFIG",
    "HUNYUAN_CONFIG",
    "get_provider_config",
    "list_providers",
    "list_mvp_providers",
    "get_model_info",
    "get_default_model",
    "register_provider",
    "check_capability",
    "find_config_file",
    "load_config_file",
    "load_config",
    "get_config",
    "clear_cache",
    "apply_to_registry",
    "ConfigError",
    "get_env_overrides",
    "apply_env_overrides_to_config",
    "apply_env_overrides",
]
