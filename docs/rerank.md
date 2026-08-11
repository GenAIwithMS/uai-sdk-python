# Rerank

## Overview

Ranks documents by relevance to a query. Currently supported by: Qwen, GLM.

## Usage

```python
from uai import UniversalAI

client = UniversalAI(api_key="...", provider="qwen")

result = client.rerank(
    query="What is quantum computing?",
    documents=["doc1...", "doc2...", "doc3..."],
    model="qwen3-rerank",
)

for item in result.results:
    print(f"  Rank {item.index}: {item.score}")
```

The request is routed through the provider's adapter to the provider-specific
rerank endpoint. Results are ordered by descending relevance.