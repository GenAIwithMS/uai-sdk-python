"""Unit tests for the Unified Data Models (Sub-module 1.1.1).

These tests verify the strict Pydantic validation, input normalization,
and field-level constraints on all model classes in ``uai.models``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from uai.models import (
    ChatMessage,
    ContentBlock,
    FinishReason,
    FunctionCall,
    FunctionDefinition,
    ImageContent,
    ImageURL,
    Role,
    StreamChunk,
    TextContent,
    ToolCall,
    ToolCallMode,
    ToolDefinition,
    UnifiedRequest,
    UnifiedResponse,
    UsageMetrics,
)

# ---------------------------------------------------------------------------
# Role & FinishReason enums
# ---------------------------------------------------------------------------


class TestRoleEnum:
    def test_all_members(self):
        assert {e.name for e in Role} == {
            "SYSTEM",
            "USER",
            "ASSISTANT",
            "TOOL",
            "DEVELOPER",
            "FUNCTION",
        }

    def test_string_values(self):
        assert Role.USER.value == "user"
        assert Role.SYSTEM.value == "system"
        assert Role.ASSISTANT.value == "assistant"

    def test_enum_from_string(self):
        assert Role("user") is Role.USER
        assert Role("assistant") is Role.ASSISTANT


class TestFinishReasonEnum:
    def test_all_members(self):
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_CALLS.value == "tool_calls"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"

    def test_from_string(self):
        assert FinishReason("stop") is FinishReason.STOP
        assert FinishReason("tool_calls") is FinishReason.TOOL_CALLS


class TestToolCallModeEnum:
    def test_values(self):
        assert ToolCallMode.NONE.value == "none"
        assert ToolCallMode.AUTO.value == "auto"
        assert ToolCallMode.REQUIRED.value == "required"


# ---------------------------------------------------------------------------
# UsageMetrics
# ---------------------------------------------------------------------------


class TestUsageMetrics:
    def test_defaults(self):
        usage = UsageMetrics()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cache_read_tokens is None
        assert usage.cache_write_tokens is None
        assert usage.reasoning_tokens is None

    def test_auto_compute_total(self):
        usage = UsageMetrics(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_explicit_total_preserved(self):
        usage = UsageMetrics(input_tokens=100, output_tokens=50, total_tokens=200)
        assert usage.total_tokens == 200

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            UsageMetrics(input_tokens=-1)

    def test_add_merge(self):
        u1 = UsageMetrics(input_tokens=100, output_tokens=50)
        u2 = UsageMetrics(input_tokens=200, output_tokens=75, cache_read_tokens=30)
        result = u1.add(u2)
        assert result.input_tokens == 300
        assert result.output_tokens == 125
        assert result.total_tokens == 425
        assert result.cache_read_tokens == 30

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            UsageMetrics(input_tokens=1, bogus=True)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TestTextContent:
    def test_valid(self):
        c = TextContent(text="Hello, world!")
        assert c.type == "text"
        assert c.text == "Hello, world!"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            TextContent(text="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            TextContent(text="hi", bogus=True)


class TestImageURL:
    def test_default_detail(self):
        url = ImageURL(url="data:image/png;base64,abc")
        assert url.detail == "auto"

    def test_invalid_detail_rejected(self):
        with pytest.raises(ValidationError):
            ImageURL(url="https://example.com/img.png", detail="ultra")

    def test_empty_url_rejected(self):
        with pytest.raises(ValidationError):
            ImageURL(url="")


class TestImageContent:
    def test_valid(self):
        c = ImageContent(
            image_url=ImageURL(url="data:image/png;base64,abc", detail="high")
        )
        assert c.type == "image_url"
        assert c.image_url.url.startswith("data:")
        assert c.image_url.detail == "high"


class TestContentBlockDiscriminatedUnion:
    def test_text_block(self):
        block: ContentBlock = TextContent(text="hello")
        assert block.type == "text"

    def test_image_block(self):
        block: ContentBlock = ImageContent(
            image_url=ImageURL(url="https://example.com/img.png")
        )
        assert block.type == "image_url"

    def test_from_dict_text(self):
        ta = TypeAdapter(ContentBlock)
        block = ta.validate_python({"type": "text", "text": "hello"})
        assert isinstance(block, TextContent)

    def test_from_dict_image(self):
        ta = TypeAdapter(ContentBlock)
        block = ta.validate_python(
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        )
        assert isinstance(block, ImageContent)


# ---------------------------------------------------------------------------
# FunctionCall & ToolCall
# ---------------------------------------------------------------------------


class TestFunctionCall:
    def test_basic(self):
        fc = FunctionCall(name="get_weather", arguments='{"city": "Beijing"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city": "Beijing"}'

    def test_empty_arguments(self):
        fc = FunctionCall(name="noop", arguments="")
        assert fc.arguments == ""

    def test_get_arguments_parsed(self):
        fc = FunctionCall(name="get_weather", arguments='{"city": "Beijing"}')
        tool_call = ToolCall(id="call_123", function=fc)
        args = tool_call.get_arguments()
        assert args == {"city": "Beijing"}

    def test_get_arguments_empty(self):
        fc = FunctionCall(name="noop", arguments="")
        tool_call = ToolCall(id="call_1", function=fc)
        assert tool_call.get_arguments() == {}

    def test_get_arguments_strips_whitespace(self):
        fc = FunctionCall(name="f", arguments='  {"a": 1}  ')
        tool_call = ToolCall(id="call_1", function=fc)
        assert tool_call.get_arguments() == {"a": 1}

    def test_get_arguments_invalid_json_raises(self):
        fc = FunctionCall(name="f", arguments="{not json}")
        tool_call = ToolCall(id="call_1", function=fc)
        with pytest.raises(json.JSONDecodeError):
            tool_call.get_arguments()


class TestToolCall:
    def test_valid(self):
        tc = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"city": "Beijing"}'),
        )
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            ToolCall(
                id="",
                function=FunctionCall(name="f", arguments="{}"),
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ToolCall(
                id="call_1",
                function=FunctionCall(name="f", arguments="{}"),
                extra_field="oops",
            )


# ---------------------------------------------------------------------------
# FunctionDefinition & ToolDefinition
# ---------------------------------------------------------------------------


class TestFunctionDefinition:
    def test_minimal(self):
        f = FunctionDefinition(name="get_weather")
        assert f.name == "get_weather"
        assert f.description is None
        assert f.parameters == {"type": "object", "properties": {}}
        assert f.strict is None

    def test_full(self):
        f = FunctionDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            strict=True,
        )
        assert f.strict is True
        assert f.parameters["required"] == ["city"]

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            FunctionDefinition(name="")


class TestToolDefinition:
    def test_from_dict_matching_docs(self):
        tool_dict = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        tool = ToolDefinition(**tool_dict)
        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert tool.function.parameters["properties"]["city"]["type"] == "string"


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_simple_text_user(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role is Role.USER
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_role_enum(self):
        msg = ChatMessage(role=Role.SYSTEM, content="You are helpful.")
        assert msg.role is Role.SYSTEM

    def test_from_dict(self):
        msg = ChatMessage(**{"role": "user", "content": "Hi"})
        assert msg.role is Role.USER
        assert msg.content == "Hi"

    def test_assistant_with_tool_calls(self):
        msg = ChatMessage(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="get_weather", arguments='{"city": "Beijing"}'),
                )
            ],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_tool_message_with_call_id(self):
        msg = ChatMessage(
            role=Role.TOOL,
            content="sunny, 25°C",
            tool_call_id="call_1",
            name="get_weather",
        )
        assert msg.tool_call_id == "call_1"
        assert msg.content == "sunny, 25°C"

    def test_tool_message_content_defaults_to_empty(self):
        msg = ChatMessage(role=Role.TOOL, tool_call_id="call_1")
        assert msg.content == ""

    def test_multimodal_content(self):
        msg = ChatMessage(
            role=Role.USER,
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(image_url=ImageURL(url="data:image/png;base64,abc")),
            ],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="hi", bogus=True)

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid_role", content="hi")


# ---------------------------------------------------------------------------
# UnifiedRequest
# ---------------------------------------------------------------------------


class TestUnifiedRequest:
    def test_minimal(self):
        req = UnifiedRequest(messages=[{"role": "user", "content": "Hello"}])
        assert req.stream is False
        assert req.model is None
        assert req.tools is None
        assert req.output_schema is None
        assert isinstance(req.messages[0], ChatMessage)
        assert req.messages[0].content == "Hello"

    def test_with_model_and_provider(self):
        req = UnifiedRequest(
            provider="deepseek",
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.provider == "deepseek"
        assert req.model == "deepseek-chat"

    def test_string_message_normalized(self):
        """A bare string in messages should be coerced to a user ChatMessage."""
        req = UnifiedRequest(messages=["Tell me a joke"])
        assert req.messages[0].role is Role.USER
        assert req.messages[0].content == "Tell me a joke"

    def test_dict_tools_normalized(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        req = UnifiedRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=tools,
        )
        assert req.tools is not None
        assert req.tools[0].function.name == "get_weather"

    def test_toolcall_object_normalized(self):
        tc = ToolCall(
            id="call_1",
            function=FunctionCall(name="f", arguments="{}"),
        )
        req = UnifiedRequest(
            messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "tool_calls": [tc]}],
        )
        assert req.messages[1].tool_calls[0].id == "call_1"

    def test_stop_string_wrapped_in_list(self):
        req = UnifiedRequest(
            messages=[{"role": "user", "content": "Hi"}],
            stop="END",
        )
        assert req.stop == ["END"]

    def test_stop_list_preserved(self):
        req = UnifiedRequest(
            messages=[{"role": "user", "content": "Hi"}],
            stop=["END1", "END2"],
        )
        assert req.stop == ["END1", "END2"]

    def test_temperature_bounds(self):
        # Valid range
        req = UnifiedRequest(messages=[], temperature=0.7)
        assert req.temperature == 0.7

        with pytest.raises(ValidationError):
            UnifiedRequest(messages=[], temperature=-0.1)

        with pytest.raises(ValidationError):
            UnifiedRequest(messages=[], temperature=2.1)

    def test_top_p_bounds(self):
        req = UnifiedRequest(messages=[], top_p=0.95)
        assert req.top_p == 0.95

        with pytest.raises(ValidationError):
            UnifiedRequest(messages=[], top_p=1.1)

    def test_max_tokens_positive(self):
        req = UnifiedRequest(messages=[], max_tokens=512)
        assert req.max_tokens == 512

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(ValidationError):
            UnifiedRequest(messages=[], max_tokens=0)

    def test_max_tokens_negative_rejected(self):
        with pytest.raises(ValidationError):
            UnifiedRequest(messages=[], max_tokens=-5)

    def test_tool_choice_auto_without_tools_raises(self):
        with pytest.raises(ValidationError, match="tool_choice"):
            UnifiedRequest(
                messages=[{"role": "user", "content": "Hi"}],
                tool_choice="auto",
                # tools intentionally None
            )

    def test_tool_choice_none_without_tools_ok(self):
        req = UnifiedRequest(
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice="none",
        )
        assert req.tool_choice is ToolCallMode.NONE

    def test_output_schema_accepts_pydantic_class(self):
        class MyModel(BaseModel):
            name: str
            age: int

        req = UnifiedRequest(
            messages=[{"role": "user", "content": "Hi"}],
            output_schema=MyModel,
        )
        assert req.output_schema is MyModel

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            UnifiedRequest(
                messages=[{"role": "user", "content": "Hi"}],
                rogue_field=True,
            )

    def test_metadata_default_empty(self):
        req = UnifiedRequest(messages=[{"role": "user", "content": "Hi"}])
        assert req.metadata == {}

    def test_metadata_accepts_dict(self):
        req = UnifiedRequest(
            messages=[{"role": "user", "content": "Hi"}],
            metadata={"session_id": "abc123"},
        )
        assert req.metadata == {"session_id": "abc123"}

    def test_presence_penalty_bounds(self):
        req_ok = UnifiedRequest(messages=[], presence_penalty=-1.0)
        assert req_ok.presence_penalty == -1.0

        with pytest.raises(ValidationError):
            UnifiedRequest(messages=[], presence_penalty=-3.0)

    def test_frequency_penalty_bounds(self):
        req_ok = UnifiedRequest(messages=[], frequency_penalty=1.5)
        assert req_ok.frequency_penalty == 1.5

        with pytest.raises(ValidationError):
            UnifiedRequest(messages=[], frequency_penalty=3.0)

    def test_empty_messages_allowed(self):
        req = UnifiedRequest(messages=[])
        assert req.messages == []


# ---------------------------------------------------------------------------
# UnifiedResponse
# ---------------------------------------------------------------------------


class TestUnifiedResponse:
    def test_minimal(self):
        resp = UnifiedResponse()
        assert resp.content is None
        assert resp.provider is None
        assert resp.model is None
        assert resp.finish_reason is FinishReason.STOP
        assert resp.tool_calls is None
        assert resp.parsed is None
        assert resp.raw is None
        assert resp.usage.input_tokens == 0

    def test_full_text_response(self):
        resp = UnifiedResponse(
            id="chatcmpl-123",
            provider="deepseek",
            model="deepseek-chat",
            content="Hello there!",
            finish_reason=FinishReason.STOP,
            usage=UsageMetrics(input_tokens=10, output_tokens=5),
        )
        assert resp.content == "Hello there!"
        assert resp.usage.total_tokens == 15
        assert resp.finish_reason is FinishReason.STOP

    def test_tool_call_response(self):
        resp = UnifiedResponse(
            provider="deepseek",
            model="deepseek-chat",
            content=None,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="get_weather", arguments='{"city": "Beijing"}'),
                )
            ],
        )
        assert resp.content is None
        assert resp.finish_reason is FinishReason.TOOL_CALLS
        assert len(resp.tool_calls) == 1

    def test_structured_output_response(self):
        class Summary(BaseModel):
            title: str
            word_count: int

        parsed = Summary(title="Test", word_count=100)
        resp = UnifiedResponse(
            provider="deepseek",
            content='{"title": "Test", "word_count": 100}',
            parsed=parsed,
            finish_reason=FinishReason.STOP,
        )
        assert resp.parsed is not None
        assert resp.parsed.title == "Test"
        assert resp.parsed.word_count == 100

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            UnifiedResponse(content="hi", bogus=True)


# ---------------------------------------------------------------------------
# StreamChunk
# ---------------------------------------------------------------------------


class TestStreamChunk:
    def test_minimal(self):
        chunk = StreamChunk()
        assert chunk.content is None
        assert chunk.is_final is False
        assert chunk.finish_reason is None

    def test_text_delta(self):
        chunk = StreamChunk(content="Hello", model="deepseek-chat", provider="deepseek")
        assert chunk.content == "Hello"

    def test_final_chunk_with_finish_reason(self):
        chunk = StreamChunk(
            content="",
            finish_reason=FinishReason.STOP,
            is_final=True,
            usage=UsageMetrics(input_tokens=10, output_tokens=5),
        )
        assert chunk.is_final is True
        assert chunk.finish_reason is FinishReason.STOP

    def test_ttft_on_first_chunk(self):
        chunk = StreamChunk(content="Hi", ttft_ms=24.5)
        assert chunk.ttft_ms == 24.5

    def test_tool_call_delta(self):
        chunk = StreamChunk(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="get", arguments='{"cit'),
                )
            ],
        )
        assert chunk.tool_calls[0].function.arguments == '{"cit'

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            StreamChunk(content="hi", bogus=True)


# ---------------------------------------------------------------------------
# Round-trip: request → response → chunk
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_request_with_multimodal_and_tools(self):
        """Build a request mirroring the docs/recipes in the implementation plan."""
        req = UnifiedRequest(
            provider="deepseek",
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's the weather?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            tool_choice="auto",
            stream=True,
        )

        assert req.provider == "deepseek"
        assert req.model == "deepseek-chat"
        assert len(req.messages) == 2
        assert req.messages[0].role is Role.SYSTEM
        assert req.messages[0].content == "You are a helpful assistant."
        assert isinstance(req.messages[1].content, list)
        assert isinstance(req.messages[1].content[0], TextContent)
        assert isinstance(req.messages[1].content[1], ImageContent)
        assert req.tools[0].function.name == "get_weather"
        assert req.max_tokens == 2048
        assert req.temperature == 0.7
        assert req.tool_choice is ToolCallMode.AUTO
        assert req.stream is True

    def test_response_from_request(self):
        req = UnifiedRequest(
            provider="deepseek",
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
        )
        resp = UnifiedResponse(
            provider=req.provider,
            model=req.model,
            content="Hi there!",
            finish_reason=FinishReason.STOP,
            usage=UsageMetrics(input_tokens=5, output_tokens=3),
        )
        assert resp.content == "Hi there!"
        assert resp.provider == "deepseek"
        assert resp.model == "deepseek-chat"
