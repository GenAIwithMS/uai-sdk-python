"""
Universal AI Provider SDK (uai-sdk-python)

A modular, extensible AI infrastructure layer that abstracts multiple LLM
providers behind a single, stable API with an opt-in middleware architecture.
"""

import contextlib

from uai.exceptions import (
    FeatureNotSupportedError,
    ResponseParsingError,
    UAIAuthenticationError,
    UAIError,
    UAIErrorGroup,
    UAINetworkError,
    UAIRateLimitError,
)

__version__ = "0.1.0"
__all__ = [
    "FeatureNotSupportedError",
    "ResponseParsingError",
    "UAIAuthenticationError",
    "UAIError",
    "UAIErrorGroup",
    "UAINetworkError",
    "UAIRateLimitError",
    "UniversalAI",
]

with contextlib.suppress(ImportError):
    from uai.client import UniversalAI
