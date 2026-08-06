"""
Universal AI Provider SDK (uai-sdk-python)

A modular, extensible AI infrastructure layer that abstracts multiple LLM
providers behind a single, stable API with an opt-in middleware architecture.
"""

from uai.exceptions import (
    FeatureNotSupportedError,
    UAIAuthenticationError,
    UAIError,
    UAIErrorGroup,
    UAINetworkError,
    UAIRateLimitError,
    ResponseParsingError,
)

__version__ = "0.1.0"
__all__ = [
    "UniversalAI",
    "UAIError",
    "UAIErrorGroup",
    "UAIAuthenticationError",
    "UAINetworkError",
    "UAIRateLimitError",
    "ResponseParsingError",
    "FeatureNotSupportedError",
]

try:
    from uai.client import UniversalAI
except ImportError:
    pass
