"""Unit tests for the GLM provider adapter.

These tests verify that the ``GLMAdapter`` correctly translates
a ``UnifiedRequest`` into the GLM API JSON schema, parses responses
back into ``UnifiedResponse``, handles SSE streaming, and maps HTTP
status codes onto the SDK exceptions.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from uai.adapters import GLMAdapter
from uai.exceptions import (
    ResponseParsingError,
    UAIAuthenticationError,
    UAIRateLimitError,
)
from uai.models import (
    ChatMessage,
    FinishReason,
    FunctionCall,
    Role,
    ToolCall,
    ToolCallMode,
    UnifiedRequest,
)


def make_request(**overrides) -> UnifiedRequest:
    params: dict = {
        "model": "glm-4.7",
        "messages": [ChatMessage(role=Role.USER, content="Hello")],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    params.update(overrides)
    return UnifiedRequest(**params)


class TestAuthenticate:
    def test_sets_api_key(self):
        adapter = GLMAdapter()
        adapter.authenticate({"api_key": "sk-test"})
        assert adapter._api_key == "sk-test"

    def test_accepts_bearer_token(self):
        adapter = GLMAdapter()
        adapter.authenticate({"bearer_token": "sk-bearer"})
        assert adapter._api_key == "sk-bearer"

    def test_raises_without_credentials(self):
        with pytest.raises(UAIAuthenticationError):
            GLMAdapter().authenticate({})


class TestCapabilities:
    def test_unknown_capabilities(self):
        caps = GLMAdapter().capabilities()
        assert caps["reasoning"] is True
        assert caps["rerank"] is True
        assert caps["vision"] is False


class TestFormatRequest:
    def test_basic_format(self):
        body = GLMAdapter().format_request(make_request())
        assert body["model"] == "glm-4.7"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    def test_default_model_when_none(self):
        """Falls back to the registry default rather than a private constant."""
        body = GLMAdapter().format_request(make_request(model=None))
        assert body["model"] == "glm-4.7"

    def test_stop_wrapped_in_list(self):
        body = GLMAdapter().format_request(make_request(stop="END"))
        assert body["stop"] == ["END"]

    def test_penalties_included(self):
        body = GLMAdapter().format_request(
            make_request(frequency_penalty=0.5, presence_penalty=0.2)
        )
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.2

    def test_thinking_enabled_for_glm47(self):
        body = GLMAdapter().format_request(make_request())
        assert body["thinking"] == {"type": "enabled"}

    def test_no_thinking_for_other_models(self):
        body = GLMAdapter().format_request(make_request(model="glm-5.1"))
        assert "thinking" not in body

    def test_tools_and_choice(self):
        request = make_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice=ToolCallMode.AUTO,
        )
        body = GLMAdapter().format_request(request)
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert body["tool_choice"] == "auto"

    def test_tool_message_round_trip(self):
        request = UnifiedRequest(
            model="glm-4.7",
            messages=[
                ChatMessage(role=Role.USER, content="What is the weather?"),
                ChatMessage(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=FunctionCall(name="get_weather", arguments="{}"),
                        )
                    ],
                ),
                ChatMessage(role=Role.TOOL, content="Sunny", tool_call_id="call_1"),
            ],
        )
        body = GLMAdapter().format_request(request)
        assert body["messages"][1]["tool_calls"][0]["id"] == "call_1"
        tool_msg = body["messages"][2]
        assert tool_msg["tool_call_id"] == "call_1"


class TestParseResponse:
    def test_parse_simple_response(self):
        raw = {
            "id": "chatcmpl-1",
            "model": "glm-4.7",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = GLMAdapter().parse_response(raw, make_request())
        assert result.content == "Hi!"
        assert result.finish_reason == FinishReason.STOP
        assert result.provider == "glm"
        assert result.usage.input_tokens == 5

    def test_parse_with_tool_calls(self):
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "LA"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = GLMAdapter().parse_response(raw, make_request())
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert result.tool_calls[0].get_arguments() == {"city": "LA"}

    def test_no_choices_raises(self):
        with pytest.raises(ResponseParsingError):
            GLMAdapter().parse_response({"choices": []}, make_request())

    def test_structured_output_parsed(self):
        class Summary(BaseModel):
            title: str
            count: int

        raw = {
            "choices": [
                {
                    "message": {
                        "content": '{"title": "t", "count": 3}',
                        "role": "assistant",
                    }
                }
            ]
        }
        result = GLMAdapter().parse_response(raw, make_request(output_schema=Summary))
        assert isinstance(result.parsed, Summary)
        assert result.parsed.title == "t"


class TestTranslateError:
    def test_401(self):
        err = GLMAdapter().translate_error(401, "bad key")
        assert isinstance(err, UAIAuthenticationError)

    def test_429_includes_retry(self):
        err = GLMAdapter().translate_error(429, "slow down")
        assert isinstance(err, UAIRateLimitError)
        assert err.retry_after == 5.0


class TestHandleStreaming:
    def _adapter(self):
        adapter = GLMAdapter()
        adapter.authenticate({"api_key": "sk-test"})
        return adapter

    def _fake_response(self, lines):
        class FakeResponse:
            def __init__(self, lines):
                self._lines = lines

            def iter_lines(self):
                yield from self._lines

        return FakeResponse(lines)

    def test_yields_content_chunks(self):
        lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"id":"1","choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        chunks = list(self._adapter().handle_streaming(self._fake_response(lines), make_request()))
        assert [c.content for c in chunks if c.content is not None] == [
            "Hello",
            " world",
        ]
        assert chunks[0].provider == "glm"

    def test_ttft_only_on_first_chunk(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            "data: [DONE]",
        ]
        chunks = list(self._adapter().handle_streaming(self._fake_response(lines), make_request()))
        content_chunks = [c for c in chunks if c.content is not None]
        assert content_chunks[0].ttft_ms is not None
        assert content_chunks[1].ttft_ms is None


class TestEmbeddings:
    def test_format_embed_request(self):
        adapter = GLMAdapter()
        body = adapter.format_embed_request("embedding-3", ["hello"])
        assert body == {"model": "embedding-3", "input": ["hello"]}

    def test_parse_embed_response(self):
        adapter = GLMAdapter()
        raw = {
            "data": [{"embedding": [0.7], "index": 0}],
            "usage": {"prompt_tokens": 4},
        }
        result = adapter.parse_embed_response(raw, "embedding-3")
        assert result.vectors[0].values == [0.7]
        assert result.vectors[0].dimension == 1
        assert result.provider == "glm"


class TestRerank:
    def test_format_rerank_request(self):
        adapter = GLMAdapter()
        body = adapter.format_rerank_request("rerankv3.5", "query", ["a", "b"])
        assert body == {"model": "rerankv3.5", "query": "query", "documents": ["a", "b"]}

    def test_parse_rerank_response(self):
        adapter = GLMAdapter()
        raw = {
            "results": [{"index": 1, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.8}]
        }
        result = adapter.parse_rerank_response(raw, "rerankv3.5")
        assert [r.index for r in result.results] == [0, 1]
        assert result.provider == "glm"
