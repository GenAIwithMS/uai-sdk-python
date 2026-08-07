"""
Universal AI Provider SDK (uai-sdk-python)

A modular, extensible AI infrastructure layer that abstracts multiple LLM
providers behind a single, stable API with an opt-in middleware architecture.
"""

from uai.client import UniversalAI
from uai.exceptions import (
    FeatureNotSupportedError,
    ResponseParsingError,
    UAIAuthenticationError,
    UAIError,
    UAIErrorGroup,
    UAINetworkError,
    UAIRateLimitError,
)
from uai.models import (
    ChatMessage,
    FinishReason,
    FunctionCall,
    FunctionDefinition,
    Role,
    StreamChunk,
    ToolCall,
    ToolCallMode,
    ToolDefinition,
    UnifiedRequest,
    UnifiedResponse,
    UsageMetrics,
)

__version__ = "0.1.0"
__all__ = [
    "ChatMessage",
    "FeatureNotSupportedError",
    "FinishReason",
    "FunctionCall",
    "FunctionDefinition",
    "ResponseParsingError",
    "Role",
    "StreamChunk",
    "ToolCall",
    "ToolCallMode",
    "ToolDefinition",
    "UAIAuthenticationError",
    "UAIError",
    "UAIErrorGroup",
    "UAINetworkError",
    "UAIRateLimitError",
    "UniversalAI",
    "UnifiedRequest",
    "UnifiedResponse",
    "UsageMetrics",
]
