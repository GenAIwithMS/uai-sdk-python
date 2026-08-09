"""
Universal AI Provider SDK (uai-sdk-python)

A modular, extensible AI infrastructure layer that abstracts multiple LLM
providers behind a single, stable API with an opt-in middleware architecture.
"""

from uai.client import UniversalAI
from uai.enforcer import CapabilityMatrixEnforcer
from uai.exceptions import (
    FeatureNotSupportedError,
    ResponseParsingError,
    UAIAuthenticationError,
    UAIError,
    UAIErrorGroup,
    UAINetworkError,
    UAIRateLimitError,
)
from uai.middleware import (
    BaseMiddleware,
    CacheMiddleware,
    LoggingMiddleware,
    RetryMiddleware,
    SpanRecorder,
    TracingMiddleware,
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
    "BaseMiddleware",
    "CacheMiddleware",
    "CapabilityMatrixEnforcer",
    "ChatMessage",
    "FeatureNotSupportedError",
    "FinishReason",
    "FunctionCall",
    "FunctionDefinition",
    "LoggingMiddleware",
    "ResponseParsingError",
    "RetryMiddleware",
    "Role",
    "SpanRecorder",
    "StreamChunk",
    "ToolCall",
    "ToolCallMode",
    "ToolDefinition",
    "TracingMiddleware",
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
