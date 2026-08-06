# Rerank

> Stub — Phase 2 content.

## Overview

Ranks documents by relevance to a query. Currently supported by: Qwen, GLM.

## Usage

```python
result = client.rerank(
    query="What is quantum computing?",
    documents=["doc1...", "doc2...", "doc3..."],
    model="gte-rerankqwen",
)

for item in result.results:
    print(f"  Rank {item.index}: {item.score}")
```