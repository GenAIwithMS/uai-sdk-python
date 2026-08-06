# Embeddings

> Stub — Phase 2 content.

## Overview

Returns vector embeddings for text inputs. Currently supported by: Qwen, GLM, MiniMax.

## Usage

```python
vectors = client.embed(
    text=["hello world", "how are you"],
    model="text-embedding-v4",
)

print(vectors[0].values)  # list of floats
print(vectors[0].dimension)
```

## Output

Each `EmbeddingResult` contains:
- `values: list[float]` — the embedding vector
- `dimension: int` — vector dimensionality
- `model: str` — provider + model used