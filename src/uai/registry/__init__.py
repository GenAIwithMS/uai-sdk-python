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

from .env import (
    apply_env_overrides,
    apply_env_overrides_to_config,
    get_env_overrides,
)
from .loader import (
    ConfigError,
    apply_to_registry,
    clear_cache,
    find_config_file,
    get_config,
    load_config,
    load_config_file,
)
from .providers import (
    DEEPSEEK_CONFIG,
    DOUBAO_CONFIG,
    GLM_CONFIG,
    HUNYUAN_CONFIG,
    KIMI_CONFIG,
    MINIMAX_CONFIG,
    MVP_PROVIDERS,
    PROVIDER_ORDER,
    PROVIDER_REGISTRY,
    QWEN_CONFIG,
    STEPFUN_CONFIG,
    check_capability,
    get_default_model,
    get_model_info,
    get_provider_config,
    list_mvp_providers,
    list_providers,
    register_provider,
)
from .schema import (
    AuthType,
    ProviderCapabilities,
    ProviderConfig,
    ProviderModel,
    ProviderPricing,
    RegionConfig,
)

__all__ = [
    "DEEPSEEK_CONFIG",
    "DOUBAO_CONFIG",
    "GLM_CONFIG",
    "HUNYUAN_CONFIG",
    "KIMI_CONFIG",
    "MINIMAX_CONFIG",
    "MVP_PROVIDERS",
    "PROVIDER_ORDER",
    "PROVIDER_REGISTRY",
    "QWEN_CONFIG",
    "STEPFUN_CONFIG",
    "AuthType",
    "ConfigError",
    "ProviderCapabilities",
    "ProviderConfig",
    "ProviderModel",
    "ProviderPricing",
    "RegionConfig",
    "apply_env_overrides",
    "apply_env_overrides_to_config",
    "apply_to_registry",
    "check_capability",
    "clear_cache",
    "find_config_file",
    "get_config",
    "get_default_model",
    "get_env_overrides",
    "get_model_info",
    "get_provider_config",
    "list_mvp_providers",
    "list_providers",
    "load_config",
    "load_config_file",
    "register_provider",
]
