"""Unit tests for the Capability Matrix Enforcer (Module 1.3.1).

Covers the standalone CapabilityMatrixEnforcer class and its wiring into
the UniversalAI client: chat/streaming/tools/vision/embeddings/rerank
gating that halts before any network or middleware work.
"""

from __future__ import annotations

import pytest

import uai.client as client_module
from uai import CapabilityMatrixEnforcer, UniversalAI
from uai.adapters.deepseek import DeepSeekAdapter
from uai.exceptions import FeatureNotSupportedError
from uai.models import ChatMessage, ImageContent, ImageURL, Role


class TestEnforcerUnit:
    def test_supports_basic_capabilities(self):
        enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-chat")
        assert enforcer.supports("chat") is True
        assert enforcer.supports("streaming") is True
        assert enforcer.supports("tools") is True
        assert enforcer.supports("vision") is False
        assert enforcer.supports("rerank") is False

    def test_supports_vision_model(self):
        enforcer = CapabilityMatrixEnforcer("qwen", "qwen-vl-max")
        assert enforcer.supports("vision") is True
        assert enforcer.supports("embeddings") is False

    def test_alias_resolution(self):
        enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-chat-1")
        assert enforcer.model == "deepseek-chat"
        assert enforcer.supports("chat") is True

    def test_unknown_capability_raises_value_error(self):
        enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-chat")
        with pytest.raises(ValueError, match="Unknown capability"):
            enforcer.supports("telepathy")

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="not found"):
            CapabilityMatrixEnforcer("deepseek", "no-such-model")

    def test_assert_supported_raises_with_metadata(self):
        enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-chat")
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            enforcer.assert_supported("vision")
        assert exc_info.value.feature == "vision"
        assert exc_info.value.provider == "deepseek"
        assert exc_info.value.model == "deepseek-chat"
        assert "vision" in str(exc_info.value)

    def test_assert_supported_passes_for_supported(self):
        CapabilityMatrixEnforcer("qwen", "qwen-vl-max").assert_supported("vision")

    def test_adapter_matrix_cross_check_blocks_mismatch(self):
        # Registry says qwen-vl-max supports vision, but a DeepSeek adapter
        # (vision=False) must veto it — the enforcer never fakes support.
        enforcer = CapabilityMatrixEnforcer(
            "qwen",
            "qwen-vl-max",
            adapter=DeepSeekAdapter(),
        )
        assert enforcer.supports("vision") is False
        with pytest.raises(FeatureNotSupportedError):
            enforcer.assert_supported("vision")

    def test_supported_and_unsupported_features(self):
        enforcer = CapabilityMatrixEnforcer("kimi", "kimi-k2.5")
        supported = set(enforcer.supported_features())
        assert {"chat", "streaming", "tools"} <= supported
        assert "vision" not in supported
        assert "embeddings" in enforcer.unsupported_features()

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="not registered"):
            CapabilityMatrixEnforcer("no-such-provider", "whatever")

    def test_error_carries_supported_features(self):
        enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-chat")
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            enforcer.assert_supported("vision")
        assert exc_info.value.supported_features is not None
        assert "chat" in exc_info.value.supported_features
        assert "vision" not in exc_info.value.supported_features
        assert "Supported features:" in str(exc_info.value)


class TestClientEnforcement:
    def test_chat_halts_before_network_on_unsupported_model(self, monkeypatch):
        called = {"n": 0}

        def boom(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("network must not be reached")

        monkeypatch.setattr(client_module.httpx, "post", boom)
        client = UniversalAI(api_key="k", provider="qwen")
        # text-embedding-v4 is not chat-capable.
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            client.chat(messages=[{"role": "user", "content": "hi"}], model="text-embedding-v4")
        assert exc_info.value.feature == "chat"
        assert called["n"] == 0

    def test_tools_gate_via_env_override(self, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_DISABLE_TOOLS", "true")
        called = {"n": 0}

        def boom(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("network must not be reached")

        monkeypatch.setattr(client_module.httpx, "post", boom)
        client = UniversalAI(api_key="k", provider="deepseek")
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            client.chat(
                messages=[{"role": "user", "content": "weather?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
            )
        assert exc_info.value.feature == "tools"
        assert called["n"] == 0

    def test_streaming_gate_via_env_override(self, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_DISABLE_STREAMING", "true")
        client = UniversalAI(api_key="k", provider="deepseek")
        # Raises at call time — before the generator is even created.
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            client.chat(messages=[{"role": "user", "content": "hi"}], stream=True)
        assert exc_info.value.feature == "streaming"

    def test_streaming_gate_passes_at_call_time_for_supported(self, monkeypatch):
        # The gate must pass synchronously (no error) so the generator is
        # returned; the network error would only surface on iteration.
        def fake_stream(*args, **kwargs):
            raise AssertionError("streaming transport must not be reached here")

        monkeypatch.setattr(client_module.httpx, "stream", fake_stream)
        client = UniversalAI(api_key="k", provider="deepseek")
        generator = client.chat(messages=[{"role": "user", "content": "hi"}], stream=True)
        assert hasattr(generator, "__iter__")

    def test_vision_gate_blocks_text_only_provider(self, monkeypatch):
        called = {"n": 0}

        def boom(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("network must not be reached")

        monkeypatch.setattr(client_module.httpx, "post", boom)
        client = UniversalAI(api_key="k", provider="deepseek")
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[ImageContent(image_url=ImageURL(url="data:image/png;base64,AAAA"))],
            )
        ]
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            client.chat(messages=messages)
        assert exc_info.value.feature == "vision"
        assert called["n"] == 0

    def test_vision_gate_allows_vision_model(self, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"id": "1", "choices": [{"message": {"content": "A cat"}}]}

            return FakeResponse()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        client = UniversalAI(api_key="k", provider="qwen")
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[ImageContent(image_url=ImageURL(url="https://example.com/cat.png"))],
            )
        ]
        result = client.chat(messages=messages, model="qwen-vl-max")
        assert result.content == "A cat"
        assert "qwen-vl-max" in captured["body"]["model"]

    def test_embed_and_rerank_gates(self):
        client = UniversalAI(api_key="k", provider="qwen")
        # qwen-plus is a chat model — no embeddings, no rerank.
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            client.embed(["hi"], model="qwen-plus")
        assert exc_info.value.feature == "embeddings"
        with pytest.raises(FeatureNotSupportedError) as exc_info:
            client.rerank("q", ["a"], model="qwen-plus")
        assert exc_info.value.feature == "rerank"

    def test_supports_preflight(self):
        client = UniversalAI(api_key="k", provider="deepseek")
        assert client.supports("chat") is True
        assert client.supports("vision") is False
        assert client.supports("vision", provider="qwen", model="qwen-vl-max") is True
        assert client.supports("embeddings", model="deepseek-reasoner") is False

    def test_middleware_not_invoked_when_gate_fails(self, monkeypatch):
        from uai.middleware import BaseMiddleware

        before_called = {"n": 0}

        class SpyMiddleware(BaseMiddleware):
            def before_request(self, request, context):
                before_called["n"] += 1
                return request

        client = UniversalAI(api_key="k", provider="qwen")
        client.use(SpyMiddleware())
        with pytest.raises(FeatureNotSupportedError):
            client.chat(messages=[{"role": "user", "content": "hi"}], model="text-embedding-v4")
        assert before_called["n"] == 0
