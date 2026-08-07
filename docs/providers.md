# Providers

This page mirrors the hardcoded provider registry (`src/uai/registry/providers.py`).
Capabilities are the **aggregated** matrix across all models of a provider — a
capability is `True` if *any* model advertises it (see
`ProviderConfig.capabilities`). The registry is validated at import time, so a
mis-configuration fails fast rather than at runtime.

## Capability Matrix

| Provider   | Chat | Streaming | Tools | Vision | Embeddings | Rerank | Audio |
|------------|:----:|:---------:|:-----:|:------:|:----------:|:------:|:-----:|
| DeepSeek   |  ✅  |    ✅     |  ✅  |   ❌   |     ✅     |   ❌   |   ❌  |
| Qwen       |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ✅   |   ❌  |
| GLM        |  ✅  |    ✅     |  ✅  |   ❌   |     ✅     |   ✅   |   ❌  |
| Kimi       |  ✅  |    ✅     |  ✅  |   ❌   |     ❌     |   ❌   |   ❌  |
| StepFun    |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ❌   |   ❌  |
| Doubao     |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ❌   |   ❌  |
| MiniMax    |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ❌   |   ❌  |
| Hunyuan    |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ❌   |   ❌  |

## Provider Notes

### DeepSeek
- **Name / key:** `deepseek`
- **Endpoint:** `https://api.deepseek.com/v1`
- **Auth:** Bearer token — `DEEPSEEK_API_KEY`
- **Models:** `deepseek-chat` (default), `deepseek-reasoner`
- **Special features:** Reasoning token output (`deepseek-reasoner`)
- **Unsupported:** Vision, Rerank, Audio, TTS, Transcription
- **Rate limits:** 300 RPM / 30 000 TPM

### Qwen (DashScope)
- **Name / key:** `qwen`
- **Endpoint:** `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Auth:** Bearer token — `DASHSCOPE_API_KEY`
- **Models:** `qwen-turbo`, `qwen-plus` (default), `qwen-vl-max`,
  `text-embedding-v4` (embedding), `qwen-reranker` (rerank)
- **Special features:** Multimodal (`qwen-vl-max`), embeddings, rerank
- **Unsupported:** Audio, TTS, Transcription
- **Regions:** `cn-hangzhou`, `cn-beijing`
- **Rate limits:** 150 RPM / 20 000 TPM

### GLM (Zhipu AI)
- **Name / key:** `glm`
- **Endpoint:** `https://open.bigmodel.cn/api/paas/v4`
- **Auth:** Bearer token — `BIGMODEL_API_KEY`
- **Models:** `glm-5.1`, `glm-4.7` (default), `glm-embedding` (embedding)
- **Special features:** Reasoning (`glm-4.7`), embeddings, rerank
- **Rate limits:** 200 RPM / 30 000 TPM

### Kimi (Moonshot AI)
- **Name / key:** `kimi`
- **Endpoint:** `https://api.moonshot.cn/v1`
- **Auth:** Bearer token — `MOONSHOT_API_KEY`
- **Models:** `kimi-k2.5` (default), `kimi-k1.5`
- **Unsupported:** Vision, Embeddings, Rerank, Audio
- **Rate limits:** 200 RPM / 25 000 TPM

### StepFun
- **Name / key:** `stepfun`
- **Endpoint:** `https://api.stepfun.com/v1`
- **Auth:** Bearer token — `STEPFUN_API_KEY`
- **Models:** `stepfun-2.5` (default), `stepfun-vision`
- **Special features:** Vision (`stepfun-vision`)
- **Rate limits:** 100 RPM / 15 000 TPM

### Doubao (ByteDance)
- **Name / key:** `doubao`
- **Endpoint:** `https://ark.cn-beijing.volces.com/api/v3`
- **Auth:** Bearer token — `DOUBAO_API_KEY`
- **Models:** `doubao-pro-32k` (default), `doubao-vision`, `doubao-embedding`
- **Requires org ID** (`organization_required=True`)
- **Rate limits:** 180 RPM / 25 000 TPM

### MiniMax
- **Name / key:** `minimax`
- **Endpoint:** `https://api.minimax.chat/v1`
- **Auth:** Bearer token — `MINIMAX_API_KEY`
- **Models:** `minimax-m2.5` (default), `minimax-embedding`
- **Special features:** Vision (`minimax-m2.5`)
- **Unsupported:** Audio, TTS, Transcription (not yet implemented)
- **Rate limits:** 200 RPM / 20 000 TPM

### Hunyuan (Tencent Cloud)
- **Name / key:** `hunyuan`
- **Endpoint:** `https://api.hunyuan.cloud.tencent.com/v1`
- **Auth:** Bearer token — `HUNYUAN_API_KEY`
- **Models:** `hunyuan-turbo` (default), `hunyuan-pro`, `hunyuan-vision`,
  `hunyuan-embedding`
- **Requires org ID** (`organization_required=True`)
- **Special features:** Reasoning (`hunyuan-pro`), vision, embeddings
- **Rate limits:** 150 RPM / 20 000 TPM

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
get_default_model("deepseek")        # -> "deepseek-chat"
check_capability("deepseek", "deepseek-reasoner", "reasoning")  # raises if unsupported
```