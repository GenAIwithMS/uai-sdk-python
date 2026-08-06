# Providers

> Stub — content to be written. Mirror of the registry's `ProviderCapabilities` matrix.

## Capability Matrix

| Provider   | Chat | Streaming | Tools | Vision | Embeddings | Rerank | Audio | Status   |
|------------|:----:|:---------:|:-----:|:------:|:----------:|:------:|:-----:|----------|
| DeepSeek   |  ✅  |    ✅     |  ✅  |   ❌   |     ✅     |   ❌   |   ❌  | MVP      |
| Qwen       |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ✅   |  Partial | MVP    |
| GLM        |  ✅  |    ✅     |  ✅  |   ❌   |     ✅     |   ✅   |   ❌  | Phase 2  |
| Kimi       |  ✅  |    ✅     |  ✅  |   ❌   |     ❌     |   ❌   |   ❌  | Phase 2  |
| StepFun    |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ❌   |   ❌  | Phase 2  |
| Doubao     |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ❌   |   ❌  | Phase 2  |
| MiniMax    |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ✅   |   ✅  | Phase 2  |
| Hunyuan    |  ✅  |    ✅     |  ✅  |   ✅   |     ✅     |   ❌   |   ❌  | Phase 3  |

## Provider Notes

### DeepSeek
- **Endpoint:** `https://api.deepseek.com/v1`
- **Models:** `deepseek-chat`, `deepseek-reasoner`
- **Special features:** Reasoning token output (deepseek-reasoner)
- **Unsupported:** Vision, TTS, Rerank

### Qwen (DashScope)
- **Endpoint:** `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Models:** `qwen-turbo`, `qwen-plus`, `qwen-vl`, `qwen-embedding`
- **Special features:** Multimodal, rerank, embeddings
- **Unsupported:** Full audio suite

(Additional provider docs will be added as adapters are implemented.)