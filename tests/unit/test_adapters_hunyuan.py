"""Unit tests for the Hunyuan (Tencent Cloud) provider adapter.

These tests verify that the ``HunyuanAdapter`` correctly translates
a ``UnifiedRequest`` into the Hunyuan API JSON schema, parses responses
back into ``UnifiedResponse``, handles SSE streaming and vision content,
and maps HTTP status codes onto the SDK exceptions.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from uai.adapters import HunyuanAdapter
from uai.exceptions import (
    ResponseParsingError,
    UAIAuthenticationError,
    UAIRateLimitError,
)
from uai.models import (
    ChatMessage,
    FinishReason,
    Role,
    ToolCallMode,
    UnifiedRequest,
)


def make_request(**overrides) -> UnifiedRequest:
    params: dict = {
        "model": "hunyuan-turbo",
        "messages": [ChatMessage(role=Role.USER, content="Hello")],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    params.update(overrides)
    return UnifiedRequest(**params)


class TestAuthenticate:
    def test_sets_api_key(self):
        adapter = HunyuanAdapter()
        adapter.authenticate({"api_key": "sk-test"})
        assert adapter._api_key == "sk-test"

    def test_sets_org_id(self):
        adapter = HunyuanAdapter()
        adapter.authenticate({"api_key": "sk-test", "organization_id": "org-123"})
        assert adapter._org_id == "org-123"

    def test_raises_without_credentials(self):
        with pytest.raises(UAIAuthenticationError):
            HunyuanAdapter().authenticate({})


class TestCapabilities:
    def test_vision_supported(self):
        caps = HunyuanAdapter().capabilities()
        assert caps["chat"] is True
        assert caps["streaming"] is True
        assert caps["vision"] is True
        assert caps["audio"] is False


class TestFormatRequest:
    def test_basic_format(self):
        body = HunyuanAdapter().format_request(make_request())
        assert body["model"] == "hunyuan-turbo"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    def test_default_model_when_none(self):
        """Falls back to the registry default rather than a private constant."""
        body = HunyuanAdapter().format_request(make_request(model=None))
        assert body["model"] == "hunyuan-turbo-latest"

    def test_penalties_included(self):
        body = HunyuanAdapter().format_request(
            make_request(frequency_penalty=0.5, presence_penalty=0.2)
        )
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.2

    def test_tools_and_choice(self):
        request = make_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice=ToolCallMode.AUTO,
        )
        body = HunyuanAdapter().format_request(request)
        assert body["tool_choice"] == "auto"

    def test_vision_content_blocks(self):
        from uai.models import ImageContent, ImageURL, TextContent

        request = UnifiedRequest(
            model="hunyuan-vision",
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
        body = HunyuanAdapter().format_request(request)
        user_content = body["messages"][0]["content"]
        assert isinstance(user_content, list)
        assert user_content[1]["type"] == "image_url"
        assert user_content[1]["image_url"]["url"] == "https://example.com/img.png"


class TestParseResponse:
    def test_parse_simple_response(self):
        raw = {
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = HunyuanAdapter().parse_response(raw, make_request())
        assert result.content == "Hi!"
        assert result.finish_reason == FinishReason.STOP
        assert result.provider == "hunyuan"

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
        result = HunyuanAdapter().parse_response(raw, make_request())
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert result.tool_calls[0].get_arguments() == {"city": "LA"}

    def test_no_choices_raises(self):
        with pytest.raises(ResponseParsingError):
            HunyuanAdapter().parse_response({"choices": []}, make_request())

    def test_structured_output_parsed(self):
        class Summary(BaseModel):
            title: str

        raw = {"choices": [{"message": {"content": '{"title": "t"}', "role": "assistant"}}]}
        result = HunyuanAdapter().parse_response(raw, make_request(output_schema=Summary))
        assert isinstance(result.parsed, Summary)
        assert result.parsed.title == "t"


class TestTranslateError:
    def test_401(self):
        err = HunyuanAdapter().translate_error(401, "bad key")
        assert isinstance(err, UAIAuthenticationError)

    def test_429_includes_retry(self):
        err = HunyuanAdapter().translate_error(429, "slow down")
        assert isinstance(err, UAIRateLimitError)
        assert err.retry_after == 5.0


class TestHandleStreaming:
    def _adapter(self):
        adapter = HunyuanAdapter()
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
        assert chunks[0].provider == "hunyuan"

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
