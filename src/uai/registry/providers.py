"""
Hardcoded provider registry for the Universal AI Provider SDK.

This module contains the canonical, statically-defined configurations for
every supported LLM provider.  Each ``ProviderConfig`` is validated at
import time against the schema in ``schema.py``, so a typo or missing
field will produce an immediate error rather than a silent runtime
failure.

**The registry is a convenience, not a gate.**  Provider catalogues change
far faster than this package is released — DeepSeek retired
``deepseek-chat`` and ``deepseek-reasoner`` on 2026-07-24, mid-way through
this SDK's 0.1.x line — so every provider ships with
``allow_unknown_models=True``.  An id this file has never heard of is passed
through to the provider verbatim (see
:meth:`~uai.registry.schema.ProviderConfig.resolve_model`).  The entries
below exist to supply *metadata* (context windows, capabilities, pricing)
and sensible defaults, not to restrict what you may call.

Verification status of the data below:

* **Verified against vendor documentation** (2026-08): DeepSeek, Qwen
  (Alibaba Model Studio), Kimi (Moonshot), MiniMax, GLM (Z.ai).
* **Best-effort** — assembled from vendor changelogs and secondary sources
  because the vendor catalogue is not publicly enumerable: StepFun, Doubao,
  Hunyuan.  Ids here may lag; pass-through covers the gap.

``pricing`` is left at zero wherever a per-token rate could not be verified.
Zero means *unknown*, not free — :mod:`uai.benchmark` cost figures are only
meaningful for models with populated pricing.
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
# Construction helpers
# ---------------------------------------------------------------------------


def _chat_model(
    model_id: str,
    display_name: str,
    *,
    context_window: int,
    max_output_tokens: int,
    tools: bool = True,
    vision: bool = False,
    reasoning: bool = False,
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
    aliases: list[str] | None = None,
) -> ProviderModel:
    """Build a chat-capable :class:`ProviderModel` with streaming enabled."""
    return ProviderModel(
        id=model_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        pricing=ProviderPricing(
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        ),
        capabilities=ProviderCapabilities(
            chat=True,
            streaming=True,
            tools=tools,
            vision=vision,
            reasoning=reasoning,
        ),
        aliases=aliases or [],
    )


def _embedding_model(
    model_id: str,
    display_name: str,
    *,
    context_window: int = 8_192,
    input_cost_per_1k: float = 0.0,
    aliases: list[str] | None = None,
) -> ProviderModel:
    """Build an embeddings-only :class:`ProviderModel`."""
    return ProviderModel(
        id=model_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=1,
        pricing=ProviderPricing(input_cost_per_1k=input_cost_per_1k),
        capabilities=ProviderCapabilities(embeddings=True),
        aliases=aliases or [],
    )


def _rerank_model(
    model_id: str,
    display_name: str,
    *,
    context_window: int = 8_192,
    aliases: list[str] | None = None,
) -> ProviderModel:
    """Build a rerank-only :class:`ProviderModel`."""
    return ProviderModel(
        id=model_id,
        display_name=display_name,
        context_window=context_window,
        max_output_tokens=1,
        capabilities=ProviderCapabilities(rerank=True),
        aliases=aliases or [],
    )


# ---------------------------------------------------------------------------
# DeepSeek — verified against https://api-docs.deepseek.com (2026-08)
# ---------------------------------------------------------------------------
#
# The V4 line replaced the V3-era ids.  ``deepseek-chat`` and
# ``deepseek-reasoner`` were discontinued 2026-07-24 after a three-month
# notice; while they lived, both routed to deepseek-v4-flash (non-thinking
# and thinking mode respectively).  They are retained here as aliases so
# existing application code keeps working, resolving to the successor the
# vendor itself named.  Thinking mode is now a request parameter, not a
# separate model id.

DEEPSEEK_CONFIG = ProviderConfig(
    name="deepseek",
    display_name="DeepSeek AI",
    base_url="https://api.deepseek.com/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="DEEPSEEK_API_KEY",
    models={
        "deepseek-v4-flash": _chat_model(
            "deepseek-v4-flash",
            "DeepSeek V4 Flash",
            context_window=1_000_000,
            max_output_tokens=384_000,
            reasoning=True,
            input_cost_per_1k=0.00014,
            output_cost_per_1k=0.00028,
            aliases=[
                "deepseek-chat",
                "deepseek-chat-latest",
                "deepseek-reasoner",
                "deepseek-reasoner-latest",
            ],
        ),
        "deepseek-v4-pro": _chat_model(
            "deepseek-v4-pro",
            "DeepSeek V4 Pro",
            context_window=1_000_000,
            max_output_tokens=384_000,
            reasoning=True,
            input_cost_per_1k=0.000435,
            output_cost_per_1k=0.00087,
        ),
    },
    default_model="deepseek-v4-flash",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=300,
    rate_limit_tpm=30_000,
    documentation_url="https://api-docs.deepseek.com",
)

# ---------------------------------------------------------------------------
# Qwen / Alibaba Model Studio (DashScope)
# Verified against https://www.alibabacloud.com/help/en/model-studio/models
# ---------------------------------------------------------------------------

QWEN_CONFIG = ProviderConfig(
    name="qwen",
    display_name="Qwen (Alibaba Model Studio)",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="DASHSCOPE_API_KEY",
    models={
        "qwen3.7-max": _chat_model(
            "qwen3.7-max", "Qwen3.7 Max", context_window=262_144, max_output_tokens=65_536
        ),
        "qwen3.7-plus": _chat_model(
            "qwen3.7-plus", "Qwen3.7 Plus", context_window=131_072, max_output_tokens=32_768
        ),
        "qwen3.6-flash": _chat_model(
            "qwen3.6-flash", "Qwen3.6 Flash", context_window=131_072, max_output_tokens=32_768
        ),
        "qwen-max": _chat_model(
            "qwen-max", "Qwen Max (rolling)", context_window=131_072, max_output_tokens=32_768
        ),
        "qwen-plus": _chat_model(
            "qwen-plus", "Qwen Plus (rolling)", context_window=131_072, max_output_tokens=32_768
        ),
        "qwen-turbo": _chat_model(
            "qwen-turbo", "Qwen Turbo (rolling)", context_window=131_072, max_output_tokens=16_384
        ),
        "qwen-vl-max": _chat_model(
            "qwen-vl-max",
            "Qwen VL Max",
            context_window=131_072,
            max_output_tokens=8_192,
            vision=True,
        ),
        "text-embedding-v4": _embedding_model(
            "text-embedding-v4", "Qwen3 Text Embedding v4", context_window=8_192
        ),
        "tongyi-embedding-vision-plus": _embedding_model(
            "tongyi-embedding-vision-plus", "Tongyi Vision Embedding Plus", context_window=8_192
        ),
        "qwen3-rerank": _rerank_model("qwen3-rerank", "Qwen3 Reranker", context_window=32_768),
    },
    default_model="qwen3.7-plus",
    default_embedding_model="text-embedding-v4",
    default_rerank_model="qwen3-rerank",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=150,
    rate_limit_tpm=20_000,
    documentation_url="https://www.alibabacloud.com/help/en/model-studio/models",
    regions={
        "cn-hangzhou": RegionConfig(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "cn-beijing": RegionConfig(
            base_url="https://dashscope.cn-beijing.aliyuncs.com/compatible-mode/v1",
        ),
        "intl": RegionConfig(
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        ),
    },
)

# ---------------------------------------------------------------------------
# GLM — Zhipu AI / Z.ai open platform
# glm-4.6 verified against https://docs.z.ai/guides/llm/glm-4.6
# ---------------------------------------------------------------------------

GLM_CONFIG = ProviderConfig(
    name="glm",
    display_name="Zhipu AI GLM",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="BIGMODEL_API_KEY",
    models={
        "glm-5.2": _chat_model(
            "glm-5.2", "GLM-5.2", context_window=204_800, max_output_tokens=131_072, reasoning=True
        ),
        "glm-4.7": _chat_model(
            "glm-4.7", "GLM-4.7", context_window=204_800, max_output_tokens=131_072, reasoning=True
        ),
        "glm-4.6": _chat_model(
            "glm-4.6", "GLM-4.6", context_window=204_800, max_output_tokens=131_072
        ),
        "glm-4.6v": _chat_model(
            "glm-4.6v", "GLM-4.6V", context_window=131_072, max_output_tokens=16_384, vision=True
        ),
        "embedding-3": _embedding_model("embedding-3", "GLM Embedding-3", context_window=8_192),
        "rerankv3.5": _rerank_model("rerankv3.5", "GLM Rerank v3.5", context_window=8_192),
    },
    default_model="glm-4.7",
    default_embedding_model="embedding-3",
    default_rerank_model="rerankv3.5",
    api_version="v4",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=200,
    rate_limit_tpm=30_000,
    documentation_url="https://docs.z.ai",
    regions={
        "cn": RegionConfig(base_url="https://open.bigmodel.cn/api/paas/v4"),
        "intl": RegionConfig(base_url="https://api.z.ai/api/paas/v4"),
    },
)

# ---------------------------------------------------------------------------
# Kimi — Moonshot AI
# Verified against https://platform.kimi.ai/docs/api/chat
# ---------------------------------------------------------------------------

KIMI_CONFIG = ProviderConfig(
    name="kimi",
    display_name="Kimi (Moonshot AI)",
    base_url="https://api.moonshot.ai/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="MOONSHOT_API_KEY",
    models={
        "kimi-k3": _chat_model(
            "kimi-k3",
            "Kimi K3",
            context_window=262_144,
            max_output_tokens=32_768,
            reasoning=True,
            aliases=["kimi-latest"],
        ),
        "kimi-k2.7-code": _chat_model(
            "kimi-k2.7-code", "Kimi K2.7 Code", context_window=262_144, max_output_tokens=32_768
        ),
        "kimi-k2.7-code-highspeed": _chat_model(
            "kimi-k2.7-code-highspeed",
            "Kimi K2.7 Code (high speed)",
            context_window=262_144,
            max_output_tokens=32_768,
        ),
        "kimi-k2.6": _chat_model(
            "kimi-k2.6", "Kimi K2.6", context_window=262_144, max_output_tokens=32_768
        ),
        "kimi-k2.5": _chat_model(
            "kimi-k2.5", "Kimi K2.5", context_window=131_072, max_output_tokens=16_384
        ),
        "moonshot-v1-128k": _chat_model(
            "moonshot-v1-128k", "Moonshot v1 128K", context_window=131_072, max_output_tokens=8_192
        ),
        "moonshot-v1-32k": _chat_model(
            "moonshot-v1-32k", "Moonshot v1 32K", context_window=32_768, max_output_tokens=8_192
        ),
        "moonshot-v1-8k": _chat_model(
            "moonshot-v1-8k", "Moonshot v1 8K", context_window=8_192, max_output_tokens=4_096
        ),
        "moonshot-v1-auto": _chat_model(
            "moonshot-v1-auto",
            "Moonshot v1 (auto context)",
            context_window=131_072,
            max_output_tokens=8_192,
        ),
        "moonshot-v1-128k-vision-preview": _chat_model(
            "moonshot-v1-128k-vision-preview",
            "Moonshot v1 128K Vision",
            context_window=131_072,
            max_output_tokens=8_192,
            vision=True,
        ),
    },
    default_model="kimi-k3",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=200,
    rate_limit_tpm=25_000,
    documentation_url="https://platform.kimi.ai/docs",
)

# ---------------------------------------------------------------------------
# MiniMax — verified against https://platform.minimax.io/docs/api-reference
# Model ids are case-sensitive ("MiniMax-M3", not "minimax-m3").
# ---------------------------------------------------------------------------

MINIMAX_CONFIG = ProviderConfig(
    name="minimax",
    display_name="MiniMax",
    base_url="https://api.minimax.io/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="MINIMAX_API_KEY",
    models={
        "MiniMax-M3": _chat_model(
            "MiniMax-M3", "MiniMax M3", context_window=1_000_000, max_output_tokens=65_536
        ),
        "MiniMax-M2.7": _chat_model(
            "MiniMax-M2.7", "MiniMax M2.7", context_window=204_800, max_output_tokens=32_768
        ),
        "MiniMax-M2.7-highspeed": _chat_model(
            "MiniMax-M2.7-highspeed",
            "MiniMax M2.7 (high speed)",
            context_window=204_800,
            max_output_tokens=32_768,
        ),
        "MiniMax-M2.5": _chat_model(
            "MiniMax-M2.5", "MiniMax M2.5", context_window=204_800, max_output_tokens=32_768
        ),
        "MiniMax-M2.1": _chat_model(
            "MiniMax-M2.1", "MiniMax M2.1", context_window=204_800, max_output_tokens=32_768
        ),
        "MiniMax-M2": _chat_model(
            "MiniMax-M2", "MiniMax M2", context_window=204_800, max_output_tokens=32_768
        ),
        "embo-01": _embedding_model("embo-01", "MiniMax Embeddings", context_window=4_096),
    },
    default_model="MiniMax-M3",
    default_embedding_model="embo-01",
    api_version="v1",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=200,
    rate_limit_tpm=20_000,
    documentation_url="https://platform.minimax.io/docs",
)

# ---------------------------------------------------------------------------
# StepFun — best-effort (vendor catalogue not publicly enumerable)
# ---------------------------------------------------------------------------

STEPFUN_CONFIG = ProviderConfig(
    name="stepfun",
    display_name="StepFun",
    base_url="https://api.stepfun.com/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="STEPFUN_API_KEY",
    models={
        "step-3.7-flash": _chat_model(
            "step-3.7-flash",
            "Step 3.7 Flash",
            context_window=262_144,
            max_output_tokens=32_768,
            vision=True,
        ),
        "step-3.5-flash": _chat_model(
            "step-3.5-flash", "Step 3.5 Flash", context_window=131_072, max_output_tokens=16_384
        ),
        "step-3": _chat_model(
            "step-3",
            "Step 3",
            context_window=65_536,
            max_output_tokens=16_384,
            vision=True,
            reasoning=True,
        ),
    },
    default_model="step-3.7-flash",
    api_version="v1",
    timeout=30.0,
    max_retries=3,
    rate_limit_rpm=100,
    rate_limit_tpm=15_000,
    documentation_url="https://platform.stepfun.com/docs",
    regions={
        "cn": RegionConfig(base_url="https://api.stepfun.com/v1"),
        "intl": RegionConfig(base_url="https://api.stepfun.ai/v1"),
    },
)

# ---------------------------------------------------------------------------
# Doubao — ByteDance Volcengine Ark. Best-effort.
# The Lite/Pro classics and the 1.5 thinking variants were retired; the
# current line is Seed 1.6 / Seed 2.0.
# ---------------------------------------------------------------------------

DOUBAO_CONFIG = ProviderConfig(
    name="doubao",
    display_name="Doubao (ByteDance Volcengine Ark)",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="ARK_API_KEY",
    models={
        "doubao-seed-2-0-pro": _chat_model(
            "doubao-seed-2-0-pro",
            "Doubao Seed 2.0 Pro",
            context_window=262_144,
            max_output_tokens=32_768,
            reasoning=True,
        ),
        "doubao-seed-2-0-code": _chat_model(
            "doubao-seed-2-0-code",
            "Doubao Seed 2.0 Code",
            context_window=262_144,
            max_output_tokens=32_768,
        ),
        "doubao-seed-1-8": _chat_model(
            "doubao-seed-1-8", "Doubao Seed 1.8", context_window=262_144, max_output_tokens=32_768
        ),
        "doubao-seed-1-6": _chat_model(
            "doubao-seed-1-6", "Doubao Seed 1.6", context_window=262_144, max_output_tokens=32_768
        ),
        "doubao-seed-1-6-vision": _chat_model(
            "doubao-seed-1-6-vision",
            "Doubao Seed 1.6 Vision",
            context_window=262_144,
            max_output_tokens=32_768,
            vision=True,
        ),
        "doubao-embedding-vision-251215": _embedding_model(
            "doubao-embedding-vision-251215",
            "Doubao Vision Embedding",
            context_window=8_192,
        ),
    },
    default_model="doubao-seed-2-0-pro",
    default_embedding_model="doubao-embedding-vision-251215",
    api_version="v3",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=180,
    rate_limit_tpm=25_000,
    documentation_url="https://www.volcengine.com/docs/82379",
    organization_required=True,
)

# ---------------------------------------------------------------------------
# Hunyuan — Tencent Cloud. Best-effort.
# ---------------------------------------------------------------------------

HUNYUAN_CONFIG = ProviderConfig(
    name="hunyuan",
    display_name="Hunyuan (Tencent Cloud)",
    base_url="https://api.hunyuan.cloud.tencent.com/v1",
    auth_type=AuthType.BEARER_TOKEN,
    api_key_env_var="HUNYUAN_API_KEY",
    models={
        "hunyuan-t1-latest": _chat_model(
            "hunyuan-t1-latest",
            "Hunyuan T1",
            context_window=262_144,
            max_output_tokens=16_384,
            reasoning=True,
        ),
        "hunyuan-turbo-latest": _chat_model(
            "hunyuan-turbo-latest",
            "Hunyuan Turbo",
            context_window=131_072,
            max_output_tokens=8_192,
        ),
        "hunyuan-pro": _chat_model(
            "hunyuan-pro", "Hunyuan Pro", context_window=262_144, max_output_tokens=8_192
        ),
        "hunyuan-vision": _chat_model(
            "hunyuan-vision",
            "Hunyuan Vision",
            context_window=131_072,
            max_output_tokens=8_192,
            vision=True,
        ),
        "hunyuan-embedding": _embedding_model(
            "hunyuan-embedding", "Hunyuan Embedding", context_window=8_192
        ),
    },
    default_model="hunyuan-turbo-latest",
    default_embedding_model="hunyuan-embedding",
    api_version="v1",
    timeout=45.0,
    max_retries=3,
    rate_limit_rpm=150,
    rate_limit_tpm=20_000,
    documentation_url="https://cloud.tencent.com/document/product/1729",
    organization_required=True,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ``PROVIDER_REGISTRY`` is a *mutable* dict so that the config-file loader can
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


def find_providers_for_model(
    model_id: str,
    registry: dict[str, ProviderConfig] | None = None,
) -> list[str]:
    """
    Return the names of every registered provider that knows *model_id*.

    Powers provider inference, so ``UniversalAI(model="glm-4.7")`` can route
    without the caller naming the provider.  An empty list means no provider
    declares the id — which is not the same as the id being invalid, since
    unregistered ids are passed through.

    :param model_id: Model id or alias (case-sensitive, as providers are).
    :param registry: Registry to search.  Defaults to ``PROVIDER_REGISTRY``.
    :return: Matching provider names in :data:`PROVIDER_ORDER` sequence.
    """
    source = PROVIDER_REGISTRY if registry is None else registry
    ordered = [name for name in PROVIDER_ORDER if name in source]
    ordered += [name for name in source if name not in ordered]
    return [name for name in ordered if source[name].knows_model(model_id)]


def register_provider(config: ProviderConfig, override: bool = False) -> ProviderConfig:
    """
    Dynamically register an additional provider at runtime.

    This is used by the config-file loader to merge user-supplied providers
    with the built-in set.

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
