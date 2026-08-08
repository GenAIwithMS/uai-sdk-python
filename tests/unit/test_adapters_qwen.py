"""Unit tests for the Qwen provider adapter (Sub-module 1.2.3).

These tests verify that the ``QwenAdapter`` correctly translates
a ``UnifiedRequest`` into the Qwen/DashScope API JSON schema, parses
responses back into ``UnifiedResponse``, handles SSE streaming and
vision content, and maps HTTP status codes onto the SDK exceptions.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from uai.adapters import QwenAdapter
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
        "model": "qwen-plus",
        "messages": [ChatMessage(role=Role.USER, content="Hello")],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    params.update(overrides)
    return UnifiedRequest(**params)


class TestAuthenticate:
    def test_sets_api_key(self):
        adapter = QwenAdapter()
        adapter.authenticate({"api_key": "sk-test"})
        assert adapter._api_key == "sk-test"

    def test_accepts_bearer_token(self):
        adapter = QwenAdapter()
        adapter.authenticate({"bearer_token": "sk-bearer"})
        assert adapter._api_key == "sk-bearer"

    def test_raises_without_credentials(self):
        adapter = QwenAdapter()
        with pytest.raises(UAIAuthenticationError):
            adapter.authenticate({})


class TestCapabilities:
    def test_multimodal_and_rerank_supported(self):
        caps = QwenAdapter().capabilities()
        assert caps["chat"] is True
        assert caps["streaming"] is True
        assert caps["vision"] is True
        assert caps["rerank"] is True

    def test_text_only_capabilities(self):
        caps = QwenAdapter().capabilities()
        assert caps["audio"] is False
        assert caps["tts"] is False
        assert caps["transcription"] is False


class TestFormatRequest:
    def test_basic_format(self):
        body = QwenAdapter().format_request(make_request())
        assert body["model"] == "qwen-plus"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.5

    def test_default_model_when_none(self):
        body = QwenAdapter().format_request(make_request(model=None))
        assert body["model"] == "qwen-plus"

    def test_stop_wrapped_in_list(self):
        body = QwenAdapter().format_request(make_request(stop="END"))
        assert body["stop"] == ["END"]

    def test_generation_params_omitted_when_none(self):
        body = QwenAdapter().format_request(
            make_request(temperature=None, max_tokens=None, presence_penalty=None)
        )
        assert "temperature" not in body
        assert "max_tokens" not in body

    def test_penalties_included(self):
        body = QwenAdapter().format_request(
            make_request(frequency_penalty=0.5, presence_penalty=0.2)
        )
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.2

    def test_tools_and_choice(self):
        request = make_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice=ToolCallMode.AUTO,
        )
        body = QwenAdapter().format_request(request)
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert body["tool_choice"] == "auto"

    def test_tool_message_round_trip(self):
        request = UnifiedRequest(
            model="qwen-plus",
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
        body = QwenAdapter().format_request(request)
        assert body["messages"][1]["tool_calls"][0]["id"] == "call_1"
        tool_msg = body["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"

    def test_vision_content_blocks(self):
        from uai.models import ImageContent, ImageURL, TextContent

        request = UnifiedRequest(
            model="qwen-vl-max",
            messages=[
                ChatMessage(
                    role=Role.USER,
                    content=[
                        TextContent(text="describe this"),
                        ImageContent(image_url=ImageURL(url="https://example.com/img.png")),
                    ],
                )
            ],
        )
        body = QwenAdapter().format_request(request)
        user_content = body["messages"][0]["content"]
        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image_url"
        assert user_content[1]["image_url"]["url"] == "https://example.com/img.png"


class TestParseResponse:
    def test_parse_simple_response(self):
        raw = {
            "id": "chatcmpl-1",
            "model": "qwen-plus",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = QwenAdapter().parse_response(raw, make_request())
        assert result.content == "Hi!"
        assert result.finish_reason == FinishReason.STOP
        assert result.provider == "qwen"
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 2

    def test_parse_dashscope_usage_names(self):
        raw = {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"input_tokens": 7, "output_tokens": 9},
        }
        result = QwenAdapter().parse_response(raw, make_request())
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 9

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
        result = QwenAdapter().parse_response(raw, make_request())
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert result.tool_calls is not None
        assert result.tool_calls[0].get_arguments() == {"city": "LA"}

    def test_content_filter_finish_reason(self):
        result = QwenAdapter().parse_response(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "content_filter"}]},
            make_request(),
        )
        assert result.finish_reason == FinishReason.CONTENT_FILTER

    def test_unknown_finish_reason_maps_to_other(self):
        result = QwenAdapter().parse_response(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "weird"}]},
            make_request(),
        )
        assert result.finish_reason == FinishReason.OTHER

    def test_no_choices_raises(self):
        with pytest.raises(ResponseParsingError):
            QwenAdapter().parse_response({"choices": []}, make_request())

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
        result = QwenAdapter().parse_response(raw, make_request(output_schema=Summary))
        assert isinstance(result.parsed, Summary)
        assert result.parsed.title == "t"

    def test_structured_invalid_json_raises(self):
        class Summary(BaseModel):
            title: str

        raw = {"choices": [{"message": {"content": "not json", "role": "assistant"}}]}
        with pytest.raises(ResponseParsingError):
            QwenAdapter().parse_response(raw, make_request(output_schema=Summary))


class TestTranslateError:
    def test_401(self):
        err = QwenAdapter().translate_error(401, "bad key")
        assert isinstance(err, UAIAuthenticationError)

    def test_429_includes_retry(self):
        err = QwenAdapter().translate_error(429, "slow down")
        assert isinstance(err, UAIRateLimitError)
        assert err.retry_after == 5.0

    def test_5xx_generic(self):
        err = QwenAdapter().translate_error(500, "boom")
        assert isinstance(err, Exception)


class TestHandleStreaming:
    def _dse(self):
        adapter = QwenAdapter()
        adapter.authenticate({"api_key": "sk-test"})
        return adapter

    def _fake_response(self, lines):
        class FakeResponse:
            def __init__(self, lines):
                self._lines = lines

            def iter_lines(self):
                yield from self._lines

        return FakeResponse(lines)

    def test_yields_content_chunks_with_metadata(self):
        adapter = self._dse()
        lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"Hello"},"index":0}]}',
            'data: {"id":"1","choices":[{"delta":{"content":" world"},"index":0}]}',
            "data: [DONE]",
        ]
        chunks = list(adapter.handle_streaming(self._fake_response(lines), make_request()))
        contents = [c.content for c in chunks if c.content is not None]
        assert contents == ["Hello", " world"]
        assert chunks[0].provider == "qwen"
        assert chunks[0].model == "qwen-plus"
        assert chunks[0].id == "1"

    def test_ttft_only_on_first_chunk(self):
        adapter = self._dse()
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            "data: [DONE]",
        ]
        chunks = list(adapter.handle_streaming(self._fake_response(lines), make_request()))
        content_chunks = [c for c in chunks if c.content is not None]
        assert content_chunks[0].ttft_ms is not None
        assert content_chunks[1].ttft_ms is None

    def test_finish_reason_marks_final(self):
        adapter = self._dse()
        lines = [
            'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        chunks = list(adapter.handle_streaming(self._fake_response(lines), make_request()))
        assert chunks[-1].is_final is True


class TestEmbeddings:
    def test_format_embed_request(self):
        adapter = QwenAdapter()
        body = adapter.format_embed_request("text-embedding-v4", ["hello", "world"])
        assert body == {"model": "text-embedding-v4", "input": ["hello", "world"]}

    def test_parse_embed_response(self):
        adapter = QwenAdapter()
        raw = {
            "model": "text-embedding-v4",
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
        result = adapter.parse_embed_response(raw, "text-embedding-v4")
        assert len(result.vectors) == 2
        assert result.vectors[0].values == [0.1, 0.2, 0.3]
        assert result.vectors[0].dimension == 3
        assert result.vectors[1].index == 1
        assert result.provider == "qwen"


class TestRerank:
    def test_format_rerank_request(self):
        adapter = QwenAdapter()
        body = adapter.format_rerank_request("gte-rerank", "what is ai", ["doc1", "doc2"])
        assert body == {"model": "gte-rerank", "query": "what is ai", "documents": ["doc1", "doc2"]}

    def test_parse_rerank_response_sorted_by_score(self):
        adapter = QwenAdapter()
        raw = {
            "model": "gte-rerank",
            "results": [
                {"index": 1, "relevance_score": 0.3},
                {"index": 0, "relevance_score": 0.9},
            ],
        }
        result = adapter.parse_rerank_response(raw, "gte-rerank")
        assert [r.index for r in result.results] == [0, 1]
        assert result.results[0].score == 0.9
        assert result.results[1].index == 1

    def test_parse_rerank_response_sets_provider(self):
        adapter = QwenAdapter()
        raw = {"results": [{"index": 0, "relevance_score": 0.5}]}
        result = adapter.parse_rerank_response(raw)
        assert result.provider == "qwen"
