"""
DeepSeek provider adapter implementation.

Translates between the Universal AI Provider SDK's unified models
and the DeepSeek API format (OpenAI-compatible chat completions).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from uai.adapters.base_adapter import BaseProviderAdapter
from uai.exceptions import (
    ResponseParsingError,
    UAIAuthenticationError,
    UAIError,
    UAIRateLimitError,
)
from uai.models import (
    FinishReason,
    Role,
    StreamChunk,
    UnifiedRequest,
    UnifiedResponse,
    UsageMetrics,
)
from uai.structured import parse_structured_output

_DEFAULT_MODEL = "deepseek-chat"
_REASONER_MODEL = "deepseek-reasoner"

_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.FUNCTION_CALL,
    "content_filter": FinishReason.CONTENT_FILTER,
    "null": FinishReason.NULL,
}


class DeepSeekAdapter(BaseProviderAdapter):
    """Adapter for DeepSeek AI API.

    Translates the SDK's ``UnifiedRequest`` into the DeepSeek chat
    completions JSON schema (OpenAI-compatible) and normalizes DeepSeek
    responses back into ``UnifiedResponse``.  Error translation maps
    DeepSeek HTTP status codes onto the SDK exception hierarchy.
    """

    provider_name = "deepseek"

    def __init__(self) -> None:
        self._api_key: str | None = None

    def authenticate(self, credentials: dict[str, Any]) -> None:
        """Set up authentication with the DeepSeek API."""
        self._api_key = credentials.get("api_key") or credentials.get("bearer_token")
        if not self._api_key:
            raise UAIAuthenticationError("DeepSeek API key required")

    def format_request(self, request: UnifiedRequest) -> dict[str, Any]:
        """Translate UnifiedRequest to DeepSeek API format."""
        body: dict[str, Any] = {}

        body["model"] = request.model or _DEFAULT_MODEL
        body["messages"] = self._format_messages(request.messages)

        generation = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }
        for key, value in generation.items():
            if value is not None:
                body[key] = value

        if request.stop:
            body["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

        if request.tools:
            body["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
            if request.tool_choice is not None:
                body["tool_choice"] = request.tool_choice.value

        if request.user is not None:
            body["user"] = request.user

        # DeepSeek reasoner exposes reasoning content via a dedicated param.
        if request.model == _REASONER_MODEL:
            body.setdefault("reasoning_format", "detailed")

        return body

    @staticmethod
    def _format_messages(messages: list[Any]) -> list[dict[str, Any]]:
        """Convert unified ``ChatMessage`` objects into DeepSeek wire format."""
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            msg_dict: dict[str, Any] = {"role": msg.role.value}
            if msg.content is not None:
                msg_dict["content"] = msg.content
            if msg.name:
                msg_dict["name"] = msg.name
            if msg.role == Role.ASSISTANT and msg.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg.tool_calls]
            if msg.role == Role.TOOL and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            formatted.append(msg_dict)
        return formatted

    def parse_response(self, response: dict[str, Any], request: UnifiedRequest) -> UnifiedResponse:
        """Translate DeepSeek API response to UnifiedResponse.

        :raises ResponseParsingError: If the response has no usable choices.
        """
        choices = response.get("choices") or []
        if not choices:
            raise ResponseParsingError(
                "DeepSeek response contains no 'choices'", provider="deepseek"
            )

        choice = choices[0]
        message = choice.get("message") or {}

        content = message.get("content") or None
        finish_reason = self._map_finish_reason(choice.get("finish_reason"))

        tool_calls = self._parse_tool_calls(message.get("tool_calls"))

        usage = self._parse_usage(response.get("usage") or {})

        parsed = None
        if request.output_schema is not None and content:
            parsed = self._parse_structured(content, request.output_schema)

        return UnifiedResponse(
            id=response.get("id"),
            provider=self.provider_name,
            model=response.get("model") or request.model,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
            parsed=parsed,
            raw=response,
        )

    @staticmethod
    def _map_finish_reason(raw: Any) -> FinishReason:
        return _FINISH_REASON_MAP.get(raw, FinishReason.OTHER)

    @staticmethod
    def _parse_tool_calls(raw: Any) -> list[Any] | None:
        if not raw:
            return None
        from uai.models import FunctionCall, ToolCall

        tool_calls = []
        for tc in raw:
            if tc.get("type") != "function":
                continue
            function = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    type="function",
                    function=FunctionCall(
                        name=function.get("name", ""),
                        arguments=function.get("arguments", "{}"),
                    ),
                )
            )
        return tool_calls or None

    @staticmethod
    def _parse_usage(data: dict[str, Any]) -> UsageMetrics:
        return UsageMetrics(
            input_tokens=data.get("prompt_tokens", 0),
            output_tokens=data.get("completion_tokens", 0),
            cache_read_tokens=data.get("cache_read_input_tokens"),
            cache_write_tokens=data.get("cache_creation_input_tokens"),
            reasoning_tokens=data.get("reasoning_tokens"),
        )

    @staticmethod
    def _parse_structured(content: str, schema: type[BaseModel]) -> BaseModel:
        """Parse and validate *content* against *schema* (Module 1.3.2)."""
        return parse_structured_output(content, schema, provider="deepseek")

    def handle_streaming(self, response: Any, request: UnifiedRequest) -> Iterator[StreamChunk]:
        """Parse DeepSeek SSE streaming response into ``StreamChunk`` objects.

        DeepSeek streams OpenAI-compatible ``data:`` lines on
        ``/chat/completions``.  The reasoner model additionally emits
        ``delta.reasoning_content`` before the final answer.  This yields
        a chunk per SSE event, enriched with provider/model/id metadata
        and a real ``ttft_ms`` value for the first content chunk.

        The accumulated reasoning text is stored on ``self.reasoning`` so
        callers can inspect the intermediate chain-of-thought.
        """
        start_time = time.monotonic()
        ttft_ms: float | None = None
        ttft_recorded = False
        seen_ids: set[str] = set()
        reasoning = ""
        self.reasoning = reasoning

        try:
            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8") if isinstance(line, bytes) else line

                if line_str.startswith("data: "):
                    line_str = line_str[6:]

                payload = line_str.strip()

                if not payload or payload == "[DONE]":
                    if payload == "[DONE]":
                        yield StreamChunk(
                            provider=self.provider_name,
                            model=request.model,
                            is_final=True,
                        )
                    continue

                try:
                    chunk_data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk_data.get("choices") or []
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta") or {}

                chunk_id = chunk_data.get("id")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)

                reasoning_delta = delta.get("reasoning_content")
                if reasoning_delta:
                    reasoning += reasoning_delta
                    self.reasoning = reasoning

                content = delta.get("content") or None
                if content is not None and not ttft_recorded:
                    ttft_ms = (time.monotonic() - start_time) * 1000
                    ttft_recorded = True

                finish_reason_raw = choice.get("finish_reason")
                finish_reason = (
                    self._map_finish_reason(finish_reason_raw)
                    if finish_reason_raw is not None
                    else None
                )

                tool_calls = self._parse_tool_calls(delta.get("tool_calls"))

                usage = None
                if chunk_data.get("usage"):
                    usage = self._parse_usage(chunk_data["usage"])

                yield StreamChunk(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    id=chunk_id,
                    model=request.model,
                    provider=self.provider_name,
                    usage=usage,
                    ttft_ms=ttft_ms,
                    is_final=finish_reason_raw is not None,
                    raw=chunk_data,
                )
                ttft_ms = None

                if finish_reason_raw is not None:
                    break
        except Exception as e:
            from uai.exceptions import UAINetworkError

            raise UAINetworkError(f"DeepSeek streaming failed: {e}") from e

    def translate_error(self, status_code: int, error_body: Any) -> Exception:
        """Translate DeepSeek API errors to SDK exceptions."""
        if status_code == 401:
            return UAIAuthenticationError(f"DeepSeek authentication failed: {error_body}")
        elif status_code == 429:
            return UAIRateLimitError(f"DeepSeek rate limited: {error_body}", retry_after=5.0)
        elif status_code >= 500:
            return UAIError(f"DeepSeek server error ({status_code}): {error_body}")
        return UAIError(f"DeepSeek API error ({status_code}): {error_body}")

    def capabilities(self) -> dict[str, bool]:
        """Return DeepSeek's capability matrix."""
        return {
            "chat": True,
            "streaming": True,
            "tools": True,
            "vision": False,
            "embeddings": True,
            "audio": False,
            "reasoning": True,
            "rerank": False,
            "tts": False,
            "transcription": False,
        }
