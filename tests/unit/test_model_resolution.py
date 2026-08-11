"""
Regression tests for provider/model resolution.

Every test here pins a defect that shipped in 0.1.x:

* a single client-wide default model leaking across a per-call
  ``provider=`` override;
* a constructor ``model=`` that was never validated;
* ``embed()``/``rerank()`` inheriting the *chat* default;
* model ids being a closed allowlist, so a model released after the SDK
  was unusable;
* environment and config-file overrides for the model having no effect;
* generation parameters and unknown kwargs being silently dropped.
"""

from __future__ import annotations

import pytest

import uai.client as client_module
from uai import UniversalAI
from uai.exceptions import ModelNotFoundError
from uai.registry import clear_cache


@pytest.fixture(autouse=True)
def _isolate_config_cache():
    """Keep a stray providers.yaml in the CWD from leaking between tests."""
    clear_cache()
    yield
    clear_cache()


def fake_chat_post(captured: dict):
    """Return a stub for ``httpx.post`` that records the outgoing request."""

    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "id": "resp-1",
                    "choices": [
                        {"message": {"content": "hi"}, "finish_reason": "stop"},
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        return FakeResponse()

    return _post


# ---------------------------------------------------------------------------
# Cross-provider defaults — the original defect
# ---------------------------------------------------------------------------


class TestPerProviderDefaults:
    def test_per_call_provider_override_uses_that_providers_default(self, monkeypatch):
        """
        A ``provider=`` override must not inherit the constructor provider's
        model. Previously this raised
        ``ValueError: Model 'deepseek-chat' not found for provider 'qwen'``.
        """
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-qwen")

        client = UniversalAI(api_key="sk-deepseek", provider="deepseek")
        client.chat(messages=[{"role": "user", "content": "hi"}], provider="qwen")

        assert captured["body"]["model"] == "qwen3.7-plus"
        assert "dashscope" in captured["url"]

    def test_supports_honours_the_overridden_provider(self):
        client = UniversalAI(api_key="k", provider="deepseek")
        assert client.supports("chat", provider="qwen") is True
        assert client.supports("vision", provider="qwen", model="qwen-vl-max") is True

    def test_constructor_model_is_scoped_to_its_provider(self, monkeypatch):
        """A model pinned for deepseek must not be sent to qwen."""
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-qwen")

        client = UniversalAI(api_key="k", provider="deepseek", model="deepseek-v4-pro")
        client.chat(messages=[{"role": "user", "content": "hi"}], provider="qwen")
        assert captured["body"]["model"] == "qwen3.7-plus"

        client.chat(messages=[{"role": "user", "content": "hi"}])
        assert captured["body"]["model"] == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# Per-modality defaults
# ---------------------------------------------------------------------------


class TestModalityDefaults:
    def test_embed_does_not_inherit_the_chat_default(self):
        client = UniversalAI(api_key="k", provider="qwen")
        assert client._model_for("qwen", None, "embeddings") == "text-embedding-v4"
        assert client._model_for("qwen", None, "rerank") == "qwen3-rerank"
        assert client._model_for("qwen", None, "chat") == "qwen3.7-plus"

    def test_embed_default_reaches_the_wire(self, monkeypatch):
        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"data": [{"embedding": [1.0], "index": 0}]}

            return FakeResponse()

        monkeypatch.setattr(client_module.httpx, "post", fake_post)
        UniversalAI(api_key="k", provider="qwen").embed("hello")
        assert captured["body"]["model"] == "text-embedding-v4"

    def test_falls_back_to_first_capable_model_without_explicit_default(self):
        """DeepSeek declares no embeddings model, so chat stays the fallback."""
        client = UniversalAI(api_key="k", provider="deepseek")
        assert client._model_for("deepseek", None, "embeddings") == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_model_from_another_provider_is_rejected_immediately(self):
        with pytest.raises(ModelNotFoundError) as exc:
            UniversalAI(api_key="k", provider="deepseek", model="qwen3.7-plus")
        assert "belongs to provider 'qwen'" in str(exc.value)

    def test_unknown_model_rejected_under_strict_models(self):
        with pytest.raises(ModelNotFoundError) as exc:
            UniversalAI(
                api_key="k", provider="deepseek", model="not-a-model", strict_models=True
            )
        assert "strict_models=False" in str(exc.value)

    def test_known_model_is_accepted(self):
        client = UniversalAI(api_key="k", provider="deepseek", model="deepseek-v4-pro")
        assert client._default_model == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# LangChain-style model acceptance
# ---------------------------------------------------------------------------


class TestModelPassThrough:
    def test_unregistered_model_is_forwarded_verbatim(self, monkeypatch):
        """
        ``UniversalAI(model=...)`` accepts an id the registry has never seen,
        the way ``ChatGroq(model=...)`` does. The registry supplies metadata;
        it does not gate what may be called.
        """
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))

        client = UniversalAI(api_key="k", provider="deepseek", model="deepseek-v5-omega")
        client.chat(messages=[{"role": "user", "content": "hi"}])
        assert captured["body"]["model"] == "deepseek-v5-omega"

    def test_per_call_model_may_also_be_unregistered(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))

        client = UniversalAI(api_key="k", provider="deepseek")
        client.chat(messages=[{"role": "user", "content": "hi"}], model="deepseek-v9-future")
        assert captured["body"]["model"] == "deepseek-v9-future"

    def test_env_flag_can_restore_strict_validation(self, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_ALLOW_UNKNOWN_MODELS", "false")
        with pytest.raises(ModelNotFoundError):
            UniversalAI(api_key="k", provider="deepseek", model="deepseek-v5-omega")

    def test_retired_ids_resolve_to_their_successor(self, monkeypatch):
        """``deepseek-chat`` was discontinued 2026-07-24; it maps to V4 Flash."""
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))

        client = UniversalAI(api_key="k", provider="deepseek", model="deepseek-chat")
        client.chat(messages=[{"role": "user", "content": "hi"}])
        assert captured["body"]["model"] == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------


class TestProviderInference:
    def test_provider_inferred_from_model(self):
        assert UniversalAI(api_key="k", model="glm-4.7")._default_provider == "glm"
        assert UniversalAI(api_key="k", model="kimi-k3")._default_provider == "kimi"

    def test_default_provider_when_nothing_given(self):
        assert UniversalAI(api_key="k")._default_provider == "deepseek"

    def test_unknown_model_cannot_be_inferred(self):
        with pytest.raises(ValueError, match="Cannot infer a provider"):
            UniversalAI(api_key="k", model="some-unheard-of-model")


# ---------------------------------------------------------------------------
# Environment and config-file overrides
# ---------------------------------------------------------------------------


class TestModelOverrides:
    def test_env_default_model_is_honoured(self, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_DEFAULT_MODEL", "deepseek-v4-pro")
        assert UniversalAI(api_key="k", provider="deepseek")._default_model == "deepseek-v4-pro"

    def test_env_default_model_may_be_unregistered(self, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_DEFAULT_MODEL", "deepseek-v4-flash-0731")
        client = UniversalAI(api_key="k", provider="deepseek")
        assert client._default_model == "deepseek-v4-flash-0731"

    def test_constructor_model_beats_env(self, monkeypatch):
        monkeypatch.setenv("UAI_PROVIDER_DEEPSEEK_DEFAULT_MODEL", "deepseek-v4-pro")
        client = UniversalAI(api_key="k", provider="deepseek", model="deepseek-v4-flash")
        assert client._default_model == "deepseek-v4-flash"

    def test_config_file_is_loaded_at_client_construction(self, tmp_path, monkeypatch):
        """
        A providers.yaml used to be ignored: the loader was exported but never
        called, despite the docs promising it ran at client init.
        """
        cfg = tmp_path / "providers.yaml"
        cfg.write_text(
            """
providers:
  deepseek:
    default_model: "deepseek-v4-custom"
    models:
      deepseek-v4-custom:
        id: "deepseek-v4-custom"
        display_name: "Custom"
        context_window: 128000
        max_output_tokens: 8192
        capabilities:
          chat: true
          streaming: true
"""
        )
        monkeypatch.setenv("UAI_CONFIG_PATH", str(cfg))
        clear_cache()

        client = UniversalAI(api_key="k", provider="deepseek", config_path=str(cfg))
        assert client._default_model == "deepseek-v4-custom"


# ---------------------------------------------------------------------------
# Request body fidelity
# ---------------------------------------------------------------------------


class TestRequestBody:
    def test_generation_parameters_reach_the_wire(self, monkeypatch):
        """
        ``frequency_penalty``/``presence_penalty``/``user`` were accepted by
        the request model but never serialized, so callers set them and got
        nothing.
        """
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))

        UniversalAI(api_key="k", provider="deepseek").chat(
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=0.5,
            presence_penalty=0.25,
            user="user-1",
        )
        body = captured["body"]
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.25
        assert body["user"] == "user-1"

    def test_unknown_kwarg_raises_instead_of_being_dropped(self):
        client = UniversalAI(api_key="k", provider="deepseek")
        with pytest.raises(TypeError, match="seed"):
            client.chat(messages=[{"role": "user", "content": "hi"}], seed=42)

    def test_chat_routes_through_the_provider_adapter(self, monkeypatch):
        """The chat path used to bypass adapters entirely."""
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))

        called: list = []
        client = UniversalAI(api_key="k", provider="deepseek")
        adapter = client._get_adapter("deepseek")
        original = adapter.format_request

        def spy(request):
            called.append(request.model)
            return original(request)

        monkeypatch.setattr(adapter, "format_request", spy)
        client.chat(messages=[{"role": "user", "content": "hi"}])

        assert called == ["deepseek-v4-flash"], "adapter.format_request must be used"

    def test_adapter_receives_canonical_id_not_alias(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(client_module.httpx, "post", fake_chat_post(captured))

        client = UniversalAI(api_key="k", provider="deepseek")
        client.chat(messages=[{"role": "user", "content": "hi"}], model="deepseek-chat-latest")
        assert captured["body"]["model"] == "deepseek-v4-flash"
