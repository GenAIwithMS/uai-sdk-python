from __future__ import annotations

from typing import Any

from uai.adapters.base_adapter import BaseProviderAdapter
from uai.exceptions import UAIAuthenticationError, UAIError, UAINetworkError, UAIRateLimitError
from uai.models import FinishReason, UnifiedRequest, UnifiedResponse, UsageMetrics


class StepFunAdapter(BaseProviderAdapter):
    """Adapter for StepFun API."""
    
    provider_name = "stepfun"
    
    def __init__(self):
        self._api_key: str | None = None
    
    def authenticate(self, credentials: dict[str, Any]) -> None:
        """Set up authentication with the StepFun API."""
        self._api_key = credentials.get("api_key") or credentials.get("bearer_token")
        if not self._api_key:
            raise UAIAuthenticationError("StepFun API key required")
    
    def format_request(self, request: UnifiedRequest) -> dict[str, Any]:
        """Translate UnifiedRequest to StepFun API format."""
        body: dict[str, Any] = {}
        
        body["model"] = request.model or "stepfun-2.5"
        
        messages = []
        for msg in request.messages:
            msg_dict: dict[str, Any] = {
                "role": msg.role.value,
            }
            if msg.content is not None:
                msg_dict["content"] = msg.content
            if msg.name:
                msg_dict["name"] = msg.name
            messages.append(msg_dict)
        body["messages"] = messages
        
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]
        
        if request.tools:
            body["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
            if request.tool_choice:
                body["tool_choice"] = request.tool_choice.value
        
        return body
    
    def parse_response(self, response: dict[str, Any], request: UnifiedRequest) -> UnifiedResponse:
        """Translate StepFun API response to UnifiedResponse."""
        choices = response.get("choices", [])
        if not choices:
            return UnifiedResponse(
                content="",
                usage=UsageMetrics(),
                finish_reason=FinishReason.STOP,
            )
        
        choice = choices[0]
        message = choice.get("message", {})
        
        content = message.get("content", "")
        
        # Map finish reason
        finish_reason_raw = choice.get("finish_reason", "stop")
        try:
            finish_reason = FinishReason(finish_reason_raw)
        except ValueError:
            finish_reason = FinishReason.STOP
        
        # Parse tool calls
        tool_calls = None
        if message.get("tool_calls"):
            from uai.models import FunctionCall, ToolCall
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type="function",
                    function=FunctionCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    ),
                )
                for tc in message["tool_calls"]
                if tc.get("type") == "function"
            ]
        
        # Usage
        usage_data = response.get("usage", {})
        usage = UsageMetrics(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        )
        
        return UnifiedResponse(
            id=response.get("id"),
            content=content if content else None,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
            raw=response,
        )
    
    def translate_error(self, status_code: int, error_body: Any) -> Exception:
        """Translate StepFun API errors to SDK exceptions."""
        if status_code == 401:
            return UAIAuthenticationError(f"StepFun authentication failed: {error_body}")
        elif status_code == 429:
            return UAIRateLimitError(f"StepFun rate limited: {error_body}")
        elif status_code >= 500:
            return UAINetworkError(f"StepFun server error ({status_code}): {error_body}")
        return UAIError(f"StepFun API error ({status_code}): {error_body}")
    
    def capabilities(self) -> dict[str, bool]:
        """Return StepFun's capability matrix."""
        return {
            "chat": True,
            "streaming": True,
            "tools": True,
            "vision": False,
            "embeddings": False,
            "audio": False,
            "reasoning": False,
            "rerank": False,
            "tts": False,
            "transcription": False,
        }