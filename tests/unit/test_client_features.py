"""Unit tests for the UniversalAI embed()/rerank() adapter routing (Phase 2).

These tests verify that ``UniversalAI.embed`` and ``UniversalAI.rerank``
route through the provider adapters (rather than inline raw calls), gate
on the provider model's capabilities, and normalize responses correctly.
"""

from __future__ import annotations

import pytest

import uai.client as client_module
from uai import UniversalAI
from uai.exceptions import FeatureNotSupportedError


class TestEmbedRouting:
    def test_embed_routes_through_adapter(self, monkeypatch):
        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "model": "text-embedding-v4",
                        "data": [{"embedding": [1.0, 2.0], "index": 0}],
                        "usage": {"prompt_tokens": 5},
                    }

            return FakeResponse()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        c = UniversalAI(api_key="sk-test", provider="qwen")
        result = c.embed(["hello"], model="text-embedding-v4")

        assert "/embeddings" in captured["url"]
        assert captured["body"] == {"model": "text-embedding-v4", "input": ["hello"]}
        assert result.vectors[0].values == [1.0, 2.0]
        assert result.vectors[0].dimension == 2

    def test_embed_single_string_is_wrapped(self, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"data": [{"embedding": [], "index": 0}]}

            return FakeResponse()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(api_key="k", provider="qwen")
        client.embed("single text", model="text-embedding-v4")
        assert captured["body"]["input"] == ["single text"]

    def test_chat_normalizes_dict_tools_in_request_body(self, monkeypatch):
        # Tools passed as raw dicts are normalized to ToolDefinition before
        # the request body is built (regression: dicts crashed model_dump).
        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["tools"] = json.get("tools")

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "id": "id-1",
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {},
                    }

            return FakeResponse()

        client = UniversalAI(api_key="k", provider="deepseek")
        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        # The dict is normalized to a ToolDefinition, gaining the default
        # empty parameters schema.
        assert captured["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def test_embed_raises_when_capability_missing(self, monkeypatch):
        monkeypatch.setattr(client_module.httpx, "post", lambda *a, **k: None)
        client = UniversalAI(api_key="k", provider="qwen")
        # qwen-plus is a chat model; embeddings are not supported for it.
        with pytest.raises(FeatureNotSupportedError):
            client.embed(["hi"], model="qwen-plus")


class TestRerankRouting:
    def test_rerank_routes_through_adapter(self, monkeypatch):
        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "results": [{"index": 1, "relevance_score": 0.3}],
                    }

            return FakeResponse()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(api_key="k", provider="qwen")
        result = client.rerank("q", ["a", "b"], model="qwen-reranker")

        assert "/rerank" in captured["url"]
        assert captured["body"] == {"model": "qwen-reranker", "query": "q", "documents": ["a", "b"]}
        assert result.results[0].index == 1
        assert result.provider == "qwen"

    def test_rerank_raises_on_unsupported_provider(self, monkeypatch):
        monkeypatch.setattr(client_module.httpx, "post", lambda *a, **k: None)
        client = UniversalAI(api_key="k", provider="qwen")
        with pytest.raises(FeatureNotSupportedError):
            client.rerank("q", ["a"], model="qwen-plus")
