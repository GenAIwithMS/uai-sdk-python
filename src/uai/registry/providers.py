"""
Hardcoded provider registry for the Universal AI Provider SDK.

This module contains the canonical, statically-defined configurations for
every supported LLM provider.  Each ``ProviderConfig`` is validated at
import time against the schema in ``schema.py``, so a typo or missing
field will produce an immediate error rather than a silent runtime
failure.

For MVP, only **DeepSeek** and **Qwen** are fully wired.  The remaining
six providers (GLM, Kimi, StepFun, Doubao, MiniMax, Hunyuan) are included
as pre-defined configs so that adapters can be implemented incrementally
without changing the registry interface.

"""

from __future__ import annotations

from uai.exceptions import FeatureNotSupportedError

from .schema import (
    AuthType,
    ProviderCapabilities,
    ProviderConfig,
    ProviderModel,
    ProviderPricing,
    RegionConfig,
)

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

# DeepSeek — MVP
_DEEPSEEK_CHAT = ProviderModel(
    id="deepseek-chat",
    display_name="DeepSeek Chat",
    context_window=128_000,
    max_output_tokens=32_000,
    pricing=ProviderPricing(input_cost_per_1k=0.014, output_cost_per_1k=0.028),
    capabilities=ProviderCapabilities(
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
    ),
    aliases=["deepseek-chat-1", "deepseek-chat-latest"],
)

_DEEPSEEK_REASONER = ProviderModel(
    id="deepseek-reasoner",
    display_name="DeepSeek Reasoner",
    context_window=128_000,
    max_output_tokens=32_000,
    pricing=ProviderPricing(input_cost_per_1k=0.014, output_cost_per_1k=0.028),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=True,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["deepseek-reasoner-1"],
)

DEEPSEEK_CONFIG = ProviderConfig(
    name="deepseek",
    display_name="DeepSeek AI",
    base_url="https://api.deepseek.com/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="DEEPSEEK_API_KEY",
    models={
        "deepseek-chat": _DEEPSEEK_CHAT,
        "deepseek-reasoner": _DEEPSEEK_REASONER,
    },
    default_model="deepseek-chat",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=300,
    rate_limit_tpm=30_000,
    documentation_url="https://platform.deepseek.com/docs",
)

# Qwen (DashScope) — MVP
_QWEN_TURBO = ProviderModel(
    id="qwen-turbo",
    display_name="Qwen Turbo",
    context_window=150_000,
    max_output_tokens=3_072,
    pricing=ProviderPricing(input_cost_per_1k=0.002, output_cost_per_1k=0.006),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["qwen-turbo-latest"],
)

_QWEN_PLUS = ProviderModel(
    id="qwen-plus",
    display_name="Qwen Plus",
    context_window=131_072,
    max_output_tokens=4_096,
    pricing=ProviderPricing(input_cost_per_1k=0.004, output_cost_per_1k=0.012),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["qwen-plus-latest"],
)

_QWEN_VL_MAX = ProviderModel(
    id="qwen-vl-max",
    display_name="Qwen VL Max",
    context_window=131_072,
    max_output_tokens=6_144,
    pricing=ProviderPricing(input_cost_per_1k=0.004, output_cost_per_1k=0.012),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=True,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["qwen-vl-max-latest"],
)

_QWEN_EMBEDDING = ProviderModel(
    id="text-embedding-v4",
    display_name="Qwen Text Embedding v4",
    context_window=8_192,
    max_output_tokens=1,
    pricing=ProviderPricing(input_cost_per_1k=0.0007, output_cost_per_1k=0.0),
    capabilities=ProviderCapabilities(
        chat=False,
        streaming=False,
        tools=False,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["qwen-embedding-1", "text-embedding-v3"],
)

_QWEN_RERANK = ProviderModel(
    id="qwen-reranker",
    display_name="Qwen Reranker",
    context_window=10_240,
    max_output_tokens=1,
    pricing=ProviderPricing(input_cost_per_1k=0.0007, output_cost_per_1k=0.0),
    capabilities=ProviderCapabilities(
        chat=False,
        streaming=False,
        tools=False,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=True,
        tts=False,
        transcription=False,
    ),
    aliases=["gte-rerankqwen-v2"],
)

QWEN_CONFIG = ProviderConfig(
    name="qwen",
    display_name="Qwen (DashScope)",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="DASHSCOPE_API_KEY",
    models={
        "qwen-turbo": _QWEN_TURBO,
        "qwen-plus": _QWEN_PLUS,
        "qwen-vl-max": _QWEN_VL_MAX,
        "text-embedding-v4": _QWEN_EMBEDDING,
        "qwen-reranker": _QWEN_RERANK,
    },
    default_model="qwen-plus",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=150,
    rate_limit_tpm=20_000,
    documentation_url="https://help.aliyun.com/zh/model-studio/",
    regions={
        "cn-hangzhou": RegionConfig(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "cn-beijing": RegionConfig(
            base_url="https://dashscope.cn-beijing.aliyuncs.com/compatible-mode/v1"
        ),
    },
)

# GLM — Phase 2
_GLM_5_1 = ProviderModel(
    id="glm-5.1",
    display_name="GLM-5.1",
    context_window=128_000,
    max_output_tokens=16_000,
    pricing=ProviderPricing(input_cost_per_1k=0.1, output_cost_per_1k=0.1),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=True,
        tts=False,
        transcription=False,
    ),
    aliases=["glm-5.1-flashx", "glm-5.1-z1"],
)

_GLM_4_7 = ProviderModel(
    id="glm-4.7",
    display_name="GLM-4.7",
    context_window=128_000,
    max_output_tokens=16_000,
    pricing=ProviderPricing(input_cost_per_1k=0.25, output_cost_per_1k=0.25),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=True,
        rerank=True,
        tts=False,
        transcription=False,
    ),
    aliases=["glm-4.7-flash", "glm-4.5v"],
)

_GLM_EMBEDDING = ProviderModel(
    id="glm-embedding",
    display_name="GLM Embedding",
    context_window=2_048,
    max_output_tokens=1,
    pricing=ProviderPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0),
    capabilities=ProviderCapabilities(
        chat=False,
        streaming=False,
        tools=False,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=True,
        tts=False,
        transcription=False,
    ),
    aliases=["embedding-3"],
)

GLM_CONFIG = ProviderConfig(
    name="glm",
    display_name="Zhipuan GLM",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="BIGMODEL_API_KEY",
    models={
        "glm-5.1": _GLM_5_1,
        "glm-4.7": _GLM_4_7,
        "glm-embedding": _GLM_EMBEDDING,
    },
    default_model="glm-4.7",
    api_version="v4",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=200,
    rate_limit_tpm=30_000,
    documentation_url="https://open.bigmodel.cn/dev/api",
)

# Kimi — Phase 2
_KIMI_K2_5 = ProviderModel(
    id="kimi-k2.5",
    display_name="Kimi K2.5",
    context_window=200_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.0015, output_cost_per_1k=0.006),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["kimi-k2.5-preview", "kimi-latest"],
)

_KIMI_K1_5 = ProviderModel(
    id="kimi-k1.5",
    display_name="Kimi K1.5",
    context_window=128_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.0015, output_cost_per_1k=0.006),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["moonshot-v1"],
)

KIMI_CONFIG = ProviderConfig(
    name="kimi",
    display_name="Kimi (Moonshot AI)",
    base_url="https://api.moonshot.cn/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="MOONSHOT_API_KEY",
    models={
        "kimi-k2.5": _KIMI_K2_5,
        "kimi-k1.5": _KIMI_K1_5,
    },
    default_model="kimi-k2.5",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=200,
    rate_limit_tpm=25_000,
    documentation_url="https://platform.moonshot.ai/docs",
)

# StepFun — Phase 2
_STEPFUN_2_5 = ProviderModel(
    id="stepfun-2.5",
    display_name="StepFun 2.5",
    context_window=128_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.001, output_cost_per_1k=0.003),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["step-1216"],
)

_STEPFUN_VISION = ProviderModel(
    id="stepfun-vision",
    display_name="StepFun Vision",
    context_window=128_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.002, output_cost_per_1k=0.006),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["step-vision-1216", "step-1216v"],
)

STEPFUN_CONFIG = ProviderConfig(
    name="stepfun",
    display_name="StepFun",
    base_url="https://api.stepfun.com/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="STEPFUN_API_KEY",
    models={
        "stepfun-2.5": _STEPFUN_2_5,
        "stepfun-vision": _STEPFUN_VISION,
    },
    default_model="stepfun-2.5",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=100,
    rate_limit_tpm=15_000,
    documentation_url="https://open.stepfun.com/docs",
)

# Doubao — Phase 2
_DOUBEO_PRO_32K = ProviderModel(
    id="doubao-pro-32k",
    display_name="Doubao Pro 32K",
    context_window=32_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.0007, output_cost_per_1k=0.0021),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["doubao-pro-32k-241115", "doubao-pro-32k-latest"],
)

_DOUBEO_VISION = ProviderModel(
    id="doubao-vision",
    display_name="Doubao Vision",
    context_window=128_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.002, output_cost_per_1k=0.006),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["doubao-vision-v0"],
)

_DOUBEO_EMBEDDING = ProviderModel(
    id="doubao-embedding",
    display_name="Doubao Embedding",
    context_window=8_192,
    max_output_tokens=1,
    pricing=ProviderPricing(input_cost_per_1k=0.0003, output_cost_per_1k=0.0),
    capabilities=ProviderCapabilities(
        chat=False,
        streaming=False,
        tools=False,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["doubao-embedding-text-01"],
)

DOUBAO_CONFIG = ProviderConfig(
    name="doubao",
    display_name="Doubao (ByteDance)",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="DOUBAO_API_KEY",
    models={
        "doubao-pro-32k": _DOUBEO_PRO_32K,
        "doubao-vision": _DOUBEO_VISION,
        "doubao-embedding": _DOUBEO_EMBEDDING,
    },
    default_model="doubao-pro-32k",
    api_version="v3",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=180,
    rate_limit_tpm=25_000,
    documentation_url="https://www.volcengine.com/docs",
    organization_required=True,
)

# MiniMax — Phase 2
_MINIMAX_M2_5 = ProviderModel(
    id="minimax-m2.5",
    display_name="MiniMax M2.5",
    context_window=200_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["minimax-m2.5-turbo", "minimax-v2-0"],
)

_MINIMAX_EMBEDDING = ProviderModel(
    id="minimax-embedding",
    display_name="MiniMax Embedding",
    context_window=8_192,
    max_output_tokens=1,
    pricing=ProviderPricing(input_cost_per_1k=0.0003, output_cost_per_1k=0.0),
    capabilities=ProviderCapabilities(
        chat=False,
        streaming=False,
        tools=False,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["minimax-embedding-0715"],
)

MINIMAX_CONFIG = ProviderConfig(
    name="minimax",
    display_name="MiniMax",
    base_url="https://api.minimax.chat/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="MINIMAX_API_KEY",
    models={
        "minimax-m2.5": _MINIMAX_M2_5,
        "minimax-embedding": _MINIMAX_EMBEDDING,
    },
    default_model="minimax-m2.5",
    api_version="v1",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=200,
    rate_limit_tpm=20_000,
    documentation_url="https://platform.minimaxi.chat/docs",
)

# Hunyuan — Phase 3
_HUNYUAN_TURBO = ProviderModel(
    id="hunyuan-turbo",
    display_name="Hunyuan Turbo",
    context_window=131_072,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.001, output_cost_per_1k=0.002),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["hunyuan-turbo-128k"],
)

_HUNYUAN_PRO = ProviderModel(
    id="hunyuan-pro",
    display_name="Hunyuan Pro",
    context_window=200_000,
    max_output_tokens=8_192,
    pricing=ProviderPricing(input_cost_per_1k=0.0015, output_cost_per_1k=0.003),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        audio=False,
        reasoning=True,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["hunyuan-pro-200k"],
)

_HUNYUAN_VISION = ProviderModel(
    id="hunyuan-vision",
    display_name="Hunyuan Vision",
    context_window=200_000,
    max_output_tokens=4_096,
    pricing=ProviderPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.006),
    capabilities=ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["hunyuan-vision-1120"],
)

_HUNYUAN_EMBEDDING = ProviderModel(
    id="hunyuan-embedding",
    display_name="Hunyuan Embedding",
    context_window=8_192,
    max_output_tokens=1,
    pricing=ProviderPricing(input_cost_per_1k=0.0002, output_cost_per_1k=0.0),
    capabilities=ProviderCapabilities(
        chat=False,
        streaming=False,
        tools=False,
        vision=False,
        embeddings=True,
        audio=False,
        reasoning=False,
        rerank=False,
        tts=False,
        transcription=False,
    ),
    aliases=["hunyuan-embedding-embedding"],
)

HUNYUAN_CONFIG = ProviderConfig(
    name="hunyuan",
    display_name="Hunyuan (Tencent Cloud)",
    base_url="https://api.hunyuan.cloud.tencent.com/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="HUNYUAN_API_KEY",
    models={
        "hunyuan-turbo": _HUNYUAN_TURBO,
        "hunyuan-pro": _HUNYUAN_PRO,
        "hunyuan-vision": _HUNYUAN_VISION,
        "hunyuan-embedding": _HUNYUAN_EMBEDDING,
    },
    default_model="hunyuan-turbo",
    api_version="v1",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=150,
    rate_limit_tpm=20_000,
    documentation_url="https://cloud.tencent.com/document/tracking",
    organization_required=True,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# The canonical, immutable set of provider configurations.
# ``PROVIDER_REGISTRY`` is a *mutable* dict so that Phase 2's loader can
# inject user-supplied configs at runtime, but the built-in providers are
# never removed.
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "deepseek": DEEPSEEK_CONFIG,
    "qwen": QWEN_CONFIG,
    "glm": GLM_CONFIG,
    "kimi": KIMI_CONFIG,
    "stepfun": STEPFUN_CONFIG,
    "doubao": DOUBAO_CONFIG,
    "minimax": MINIMAX_CONFIG,
    "hunyuan": HUNYUAN_CONFIG,
}

# The subset of providers targeted for the MVP release.
MVP_PROVIDERS: list[str] = ["deepseek", "qwen"]

# Ordered by roadmap phase for predictable iteration.
PROVIDER_ORDER: list[str] = [
    "deepseek",
    "qwen",
    "glm",
    "minimax",
    "kimi",
    "stepfun",
    "doubao",
    "hunyuan",
]


def get_provider_config(provider_name: str) -> ProviderConfig:
    """
    Retrieve the :class:`ProviderConfig` for *provider_name*.

    :param provider_name: Canonical provider name (case-insensitive).
    :return: Validated :class:`ProviderConfig`.
    :raises ValueError: If the provider is not registered.
    """
    key = provider_name.strip().lower()
    if key not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Provider '{provider_name}' is not registered. Available providers: {available}"
        )
    return PROVIDER_REGISTRY[key]


def list_providers() -> list[str]:
    """
    Return a list of all registered provider names, ordered by roadmap phase.
    """
    return [name for name in PROVIDER_ORDER if name in PROVIDER_REGISTRY]


def list_mvp_providers() -> list[str]:
    """Return only the providers targeted for the MVP release."""
    return [name for name in MVP_PROVIDERS if name in PROVIDER_REGISTRY]


def get_model_info(provider_name: str, model_id: str) -> ProviderModel:
    """
    Retrieve the :class:`ProviderModel` metadata for a specific provider/model pair.

    :param provider_name: Canonical provider name.
    :param model_id: Model id or alias.
    :return: Validated :class:`ProviderModel`.
    :raises ValueError: If the provider or model is not found.
    """
    config = get_provider_config(provider_name)
    return config.get_model(model_id)


def get_default_model(provider_name: str) -> str:
    """
    Return the default model id for *provider_name*.

    :param provider_name: Canonical provider name.
    :return: Default model id string.
    :raises ValueError: If the provider is not registered.
    """
    config = get_provider_config(provider_name)
    return config.default_model


def register_provider(config: ProviderConfig, override: bool = False) -> ProviderConfig:
    """
    Dynamically register an additional provider at runtime.

    This is used by Phase 2's config-file loader to merge user-supplied
    providers with the built-in set.

    :param config:   Validated :class:`ProviderConfig` to register.
    :param override: If ``True``, overwrite an existing provider with the
                     same name.  Defaults to ``False``.
    :return: The registered :class:`ProviderConfig`.
    :raises ValueError: If the provider already exists and ``override`` is
                        ``False``.
    """
    key = config.name.lower()
    if key in PROVIDER_REGISTRY and not override:
        raise ValueError(
            f"Provider '{key}' is already registered. Pass override=True to replace it."
        )
    PROVIDER_REGISTRY[key] = config
    if key not in PROVIDER_ORDER:
        PROVIDER_ORDER.append(key)
    return config


def check_capability(provider_name: str, model_id: str, capability: str) -> None:
    """
    Assert that *provider* / *model* supports *capability*.

    :param provider_name: Canonical provider name.
    :param model_id:      Model id or alias.
    :param capability:    Capability field name (e.g. ``"vision"``).
    :raises FeatureNotSupportedError: If the capability is not available.
    :raises ValueError: If the provider or model is not found.
    """
    provider = get_provider_config(provider_name)
    model = provider.get_model(model_id)

    if not hasattr(model.capabilities, capability):
        raise ValueError(
            f"Unknown capability '{capability}'. "
            f"Valid capabilities: {list(ProviderCapabilities.model_fields.keys())}"
        )

    if not getattr(model.capabilities, capability):
        raise FeatureNotSupportedError(feature=capability, provider=provider_name, model=model_id)
