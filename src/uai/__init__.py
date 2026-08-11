"""
Universal AI Provider SDK (uai-sdk-python)

A modular, extensible AI infrastructure layer that abstracts multiple LLM
providers behind a single, stable API with an opt-in middleware architecture.
"""

from uai.client import UniversalAI
from uai.enforcer import CapabilityMatrixEnforcer
from uai.exceptions import (
    FeatureNotSupportedError,
    ModelNotFoundError,
    ResponseParsingError,
    UAIAuthenticationError,
    UAICircuitOpenError,
    UAIError,
    UAIErrorGroup,
    UAINetworkError,
    UAIRateLimitError,
)
from uai.middleware import (
    BaseMiddleware,
    CacheMiddleware,
    CircuitBreakerMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MetricsRegistry,
    MiddlewareEngine,
    MiddlewareHalt,
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

__version__ = "0.2.0"
__all__ = [
    "BaseMiddleware",
    "CacheMiddleware",
    "CapabilityMatrixEnforcer",
    "ChatMessage",
    "CircuitBreakerMiddleware",
    "FeatureNotSupportedError",
    "FinishReason",
    "FunctionCall",
    "FunctionDefinition",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "MetricsRegistry",
    "MiddlewareEngine",
    "MiddlewareHalt",
    "ModelNotFoundError",
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
    "UAICircuitOpenError",
    "UAIError",
    "UAIErrorGroup",
    "UAINetworkError",
    "UAIRateLimitError",
    "UnifiedRequest",
    "UnifiedResponse",
    "UniversalAI",
    "UsageMetrics",
]
