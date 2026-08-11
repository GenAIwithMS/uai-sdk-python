# Providers

This page mirrors the provider registry (`src/uai/registry/providers.py`).
Capabilities are the **aggregated** matrix across all models of a provider — a
capability is `True` if *any* model advertises it (see
`ProviderConfig.capabilities`).

> **The registry is metadata, not an allowlist.** A model id it does not know is
> still forwarded to the provider (see
> [Using a model the registry doesn't know](configuration.md#using-a-model-the-registry-doesnt-know)).
> The entries below supply defaults, context windows and capability hints —
> they never restrict what you may call.

**Verification status (2026-08).** DeepSeek, Qwen, Kimi, MiniMax and GLM entries
were checked against vendor documentation. StepFun, Doubao and Hunyuan are
best-effort, assembled from vendor changelogs and secondary sources because
those catalogues are not publicly enumerable. `pricing` is `0.0` wherever a
per-token rate could not be verified — zero means *unknown*, not free.

## Capability Matrix

| Provider | Chat | Streaming | Tools | Vision | Embeddings | Rerank | Reasoning |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DeepSeek AI (`deepseek`) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Qwen (Alibaba Model Studio) (`qwen`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Zhipu AI GLM (`glm`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MiniMax (`minimax`) | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Kimi (Moonshot AI) (`kimi`) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| StepFun (`stepfun`) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Doubao (ByteDance Volcengine Ark) (`doubao`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Hunyuan (Tencent Cloud) (`hunyuan`) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |

## Provider Notes

### DeepSeek AI

- **Name / key:** `deepseek`
- **Endpoint:** `https://api.deepseek.com/v1`
- **Auth:** bearer_token — `DEEPSEEK_API_KEY`
- **Chat models:** `deepseek-v4-flash` *(default)*, `deepseek-v4-pro`
- **Aliases:** `deepseek-chat` → `deepseek-v4-flash`, `deepseek-chat-latest` → `deepseek-v4-flash`, `deepseek-reasoner` → `deepseek-v4-flash`, `deepseek-reasoner-latest` → `deepseek-v4-flash`
- **Rate limits:** 300 RPM / 30000 TPM

### Qwen (Alibaba Model Studio)

- **Name / key:** `qwen`
- **Endpoint:** `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Auth:** bearer_token — `DASHSCOPE_API_KEY`
- **Chat models:** `qwen3.7-max`, `qwen3.7-plus` *(default)*, `qwen3.6-flash`, `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen-vl-max`
- **Embedding models:** `text-embedding-v4` *(default)*, `tongyi-embedding-vision-plus`
- **Rerank models:** `qwen3-rerank` *(default)*
- **Regions:** `cn-hangzhou`, `cn-beijing`, `intl`
- **Rate limits:** 150 RPM / 20000 TPM

### Zhipu AI GLM

- **Name / key:** `glm`
- **Endpoint:** `https://open.bigmodel.cn/api/paas/v4`
- **Auth:** bearer_token — `BIGMODEL_API_KEY`
- **Chat models:** `glm-5.2`, `glm-4.7` *(default)*, `glm-4.6`, `glm-4.6v`
- **Embedding models:** `embedding-3` *(default)*
- **Rerank models:** `rerankv3.5` *(default)*
- **Regions:** `cn`, `intl`
- **Rate limits:** 200 RPM / 30000 TPM

### MiniMax

- **Name / key:** `minimax`
- **Endpoint:** `https://api.minimax.io/v1`
- **Auth:** bearer_token — `MINIMAX_API_KEY`
- **Chat models:** `MiniMax-M3` *(default)*, `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`, `MiniMax-M2.5`, `MiniMax-M2.1`, `MiniMax-M2`
- **Embedding models:** `embo-01` *(default)*
- **Rate limits:** 200 RPM / 20000 TPM

### Kimi (Moonshot AI)

- **Name / key:** `kimi`
- **Endpoint:** `https://api.moonshot.ai/v1`
- **Auth:** bearer_token — `MOONSHOT_API_KEY`
- **Chat models:** `kimi-k3` *(default)*, `kimi-k2.7-code`, `kimi-k2.7-code-highspeed`, `kimi-k2.6`, `kimi-k2.5`, `moonshot-v1-128k`, `moonshot-v1-32k`, `moonshot-v1-8k`, `moonshot-v1-auto`, `moonshot-v1-128k-vision-preview`
- **Aliases:** `kimi-latest` → `kimi-k3`
- **Rate limits:** 200 RPM / 25000 TPM

### StepFun

- **Name / key:** `stepfun`
- **Endpoint:** `https://api.stepfun.com/v1`
- **Auth:** bearer_token — `STEPFUN_API_KEY`
- **Chat models:** `step-3.7-flash` *(default)*, `step-3.5-flash`, `step-3`
- **Regions:** `cn`, `intl`
- **Rate limits:** 100 RPM / 15000 TPM

### Doubao (ByteDance Volcengine Ark)

- **Name / key:** `doubao`
- **Endpoint:** `https://ark.cn-beijing.volces.com/api/v3`
- **Auth:** bearer_token — `ARK_API_KEY`
- **Chat models:** `doubao-seed-2-0-pro` *(default)*, `doubao-seed-2-0-code`, `doubao-seed-1-8`, `doubao-seed-1-6`, `doubao-seed-1-6-vision`
- **Embedding models:** `doubao-embedding-vision-251215` *(default)*
- **Requires org ID** (`organization_required=True`)
- **Rate limits:** 180 RPM / 25000 TPM

### Hunyuan (Tencent Cloud)

- **Name / key:** `hunyuan`
- **Endpoint:** `https://api.hunyuan.cloud.tencent.com/v1`
- **Auth:** bearer_token — `HUNYUAN_API_KEY`
- **Chat models:** `hunyuan-t1-latest`, `hunyuan-turbo-latest` *(default)*, `hunyuan-pro`, `hunyuan-vision`
- **Embedding models:** `hunyuan-embedding` *(default)*
- **Requires org ID** (`organization_required=True`)
- **Rate limits:** 150 RPM / 20000 TPM

## Capability enforcement

Per-model capabilities are enforced at runtime by the
`CapabilityMatrixEnforcer` (Module 1.3.1). The client builds an enforcer for
the resolved provider/model at the top of every call and raises
`FeatureNotSupportedError` before any network or middleware work when a
feature is unsupported — e.g. `tools`, `streaming`, `vision` image content
on `chat()`, `embeddings` on `embed()`, `rerank` on `rerank()`.

```python
from uai import CapabilityMatrixEnforcer

enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-v4-pro")
enforcer.supports("reasoning")   # True
enforcer.assert_supported("embeddings")  # raises FeatureNotSupportedError

# Or pre-flight via the client:
client.supports("rerank", provider="qwen", model="qwen3-rerank")  # True
```

## Registry API

```python
from uai.registry import (
    get_provider_config,
    list_providers,
    list_mvp_providers,
    get_model_info,
    get_default_model,
    register_provider,
    check_capability,
)

list_providers()          # -> ordered list of all registered providers
list_mvp_providers()      # -> ["deepseek", "qwen"]
get_provider_config("qwen")          # -> ProviderConfig
get_model_info("qwen", "qwen-turbo") # -> ProviderModel (aliases supported)
get_default_model("deepseek")        # -> "deepseek-v4-flash"
check_capability("deepseek", "deepseek-v4-pro", "reasoning")  # raises if unsupported
```