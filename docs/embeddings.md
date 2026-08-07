# Embeddings

## Overview

Returns vector embeddings for text inputs. Currently supported by: DeepSeek, Qwen, GLM, StepFun, Doubao, MiniMax, Hunyuan.

## Usage

```python
from uai import UniversalAI

client = UniversalAI(api_key="...", provider="qwen")

result = client.embed(
    text=["hello world", "how are you"],
    model="text-embedding-v4",
)

print(result.vectors[0].values)  # list of floats
print(result.vectors[0].dimension)
```

`embed` accepts either a single string or a list of strings. The request is
routed through the provider's adapter to the OpenAI-compatible
`POST {base_url}/embeddings` endpoint.

## Output

Each `EmbeddingsResponse` contains:
- `vectors: list[EmbeddingResult]` — one per input text
- Each `EmbeddingResult` has:
  - `values: list[float]` — the embedding vector
  - `dimension: int` — vector dimensionality
  - `index: int` — index of the input this vector corresponds to
- `provider: str` — provider used
- `model: str` — model used
- `usage: UsageMetrics` — token usage