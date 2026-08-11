"""Unit tests for the DeepSeek provider adapter (Sub-module 1.2.2).

These tests verify that the ``DeepSeekAdapter`` correctly translates
a ``UnifiedRequest`` into the DeepSeek API JSON schema, parses responses
back into ``UnifiedResponse``, handles SSE streaming, and maps HTTP
status codes onto the SDK exception hierarchy.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from uai.adapters import DeepSeekAdapter
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
    UnifiedResponse,
)


def make_request(**overrides) -> UnifiedRequest:
    params: dict = {
        "model": "deepseek-chat",
        "messages": [ChatMessage(role=Role.USER, content="Hello")],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    params.update(overrides)
    return UnifiedRequest(**params)


class TestAuthenticate:
    def test_sets_api_key(self):
        adapter = DeepSeekAdapter()
        adapter.authenticate({"api_key": "sk-test"})
        assert adapter._api_key == "sk-test"

    def test_accepts_bearer_token(self):
        adapter = DeepSeekAdapter()
        adapter.authenticate({"bearer_token": "sk-bearer"})
        assert adapter._api_key == "sk-bearer"

    def test_raises_without_credentials(self):
        adapter = DeepSeekAdapter()
        with pytest.raises(UAIAuthenticationError):
            adapter.authenticate({})


class TestCapabilities:
    def test_chat_and_streaming_supported(self):
        caps = DeepSeekAdapter().capabilities()
        assert caps["chat"] is True
        assert caps["streaming"] is True
        assert caps["tools"] is True

    def test_vision_and_audio_unsupported(self):
        caps = DeepSeekAdapter().capabilities()
        assert caps["vision"] is False
        assert caps["audio"] is False
        assert caps["rerank"] is False

    def test_reasoning_supported(self):
        caps = DeepSeekAdapter().capabilities()
        assert caps["reasoning"] is True


class TestFormatRequest:
    def test_basic_format(self):
        body = DeepSeekAdapter().format_request(make_request())
        assert body["model"] == "deepseek-chat"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.5

    def test_default_model_when_none(self):
        """Falls back to the registry default rather than a private constant."""
        body = DeepSeekAdapter().format_request(make_request(model=None))
        assert body["model"] == "deepseek-v4-flash"

    def test_stop_wrapped_in_list(self):
        body = DeepSeekAdapter().format_request(make_request(stop="END"))
        assert body["stop"] == ["END"]

    def test_stop_list_passthrough(self):
        body = DeepSeekAdapter().format_request(make_request(stop=["A", "B"]))
        assert body["stop"] == ["A", "B"]

    def test_generation_params_omitted_when_none(self):
        body = DeepSeekAdapter().format_request(
            make_request(temperature=None, max_tokens=None, frequency_penalty=None)
        )
        assert "temperature" not in body
        assert "max_tokens" not in body

    def test_penalties_included(self):
        body = DeepSeekAdapter().format_request(
            make_request(frequency_penalty=0.5, presence_penalty=0.2)
        )
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.2

    def test_no_reasoning_param_inferred_from_model_name(self):
        """
        V4 selects thinking mode by request parameter, not by model id.

        The adapter used to inject ``reasoning_format`` whenever the model was
        ``deepseek-reasoner``; that id was retired 2026-07-24 and the branch
        would now silently attach an unsupported field to every request made
        with the legacy alias.
        """
        body = DeepSeekAdapter().format_request(make_request(model="deepseek-reasoner"))
        assert "reasoning_format" not in body
        assert body["model"] == "deepseek-reasoner"

    def test_tools_and_choice(self):
        request = make_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice=ToolCallMode.AUTO,
        )
        body = DeepSeekAdapter().format_request(request)
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert body["tool_choice"] == "auto"

    def test_tool_message_round_trip(self):
        request = UnifiedRequest(
            model="deepseek-chat",
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
        body = DeepSeekAdapter().format_request(request)
        assistant = body["messages"][1]
        assert assistant["tool_calls"][0]["id"] == "call_1"
        tool_msg = body["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"] == "Sunny"


class TestParseResponse:
    def test_parse_simple_response(self):
        raw = {
            "id": "chatcmpl-1",
            "model": "deepseek-chat",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = DeepSeekAdapter().parse_response(raw, make_request())
        assert isinstance(result, UnifiedResponse)
        assert result.content == "Hi!"
        assert result.finish_reason == FinishReason.STOP
        assert result.provider == "deepseek"
        assert result.model == "deepseek-chat"
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 2

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
        result = DeepSeekAdapter().parse_response(raw, make_request())
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert result.content is None
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].get_arguments() == {"city": "LA"}

    def test_parse_length_and_content_filter(self):
        adapter = DeepSeekAdapter()
        res_len = adapter.parse_response(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]},
            make_request(),
        )
        assert res_len.finish_reason == FinishReason.LENGTH

        res_filter = adapter.parse_response(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "content_filter"}]},
            make_request(),
        )
        assert res_filter.finish_reason == FinishReason.CONTENT_FILTER

    def test_unknown_finish_reason_maps_to_other(self):
        result = DeepSeekAdapter().parse_response(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "weird"}]},
            make_request(),
        )
        assert result.finish_reason == FinishReason.OTHER

    def test_no_choices_raises(self):
        with pytest.raises(ResponseParsingError):
            DeepSeekAdapter().parse_response({"choices": []}, make_request())

    def test_structured_output_parsed(self):
        class Summary(BaseModel):
            title: str
            count: int

        raw = {
            "choices": [
                {
                    "message": {"content": '{"title": "t", "count": 3}', "role": "assistant"},
                    "finish_reason": "stop",
                }
            ]
        }
        result = DeepSeekAdapter().parse_response(raw, make_request(output_schema=Summary))
        assert isinstance(result.parsed, Summary)
        assert result.parsed.title == "t"

    def test_structured_invalid_json_raises(self):
        class Summary(BaseModel):
            title: str

        raw = {"choices": [{"message": {"content": "not json", "role": "assistant"}}]}
        with pytest.raises(ResponseParsingError):
            DeepSeekAdapter().parse_response(raw, make_request(output_schema=Summary))

    def test_usage_reasoning_tokens(self):
        raw = {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "reasoning_tokens": 5},
        }
        result = DeepSeekAdapter().parse_response(raw, make_request())
        assert result.usage.reasoning_tokens == 5


class TestTranslateError:
    def test_401(self):
        err = DeepSeekAdapter().translate_error(401, "bad key")
        assert isinstance(err, UAIAuthenticationError)

    def test_429_includes_retry(self):
        err = DeepSeekAdapter().translate_error(429, "slow down")
        assert isinstance(err, UAIRateLimitError)
        assert err.retry_after == 5.0

    def test_5xx_generic(self):
        err = DeepSeekAdapter().translate_error(500, "boom")
        assert isinstance(err, Exception)


class TestHandleStreaming:
    def _dse(self):
        adapter = DeepSeekAdapter()
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
        assert chunks[0].provider == "deepseek"
        assert chunks[0].model == "deepseek-chat"
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
        final = chunks[-1]
        assert final.is_final is True

    def test_accumulates_reasoning_content(self):
        adapter = self._dse()
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":" more"}}]}',
            "data: [DONE]",
        ]
        list(adapter.handle_streaming(self._fake_response(lines), make_request()))
        assert adapter.reasoning == "think more"
