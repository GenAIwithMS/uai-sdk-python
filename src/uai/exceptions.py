"""
Exception hierarchy for the Universal AI Provider SDK.

All SDK-specific exceptions derive from ``UAIError``, enabling
applications to catch any SDK-related failure with a single
``except UAIError`` block.
"""

from __future__ import annotations

from typing import Any, List, Optional


class UAIError(Exception):
    """Base exception for all SDK errors."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class UAIErrorGroup(UAIError):
    """Raised when multiple errors occur (e.g. fallback across providers)."""

    def __init__(self, errors: List[UAIError], message: str = "") -> None:
        self.errors = errors
        super().__init__(message or f"Multiple errors: {len(errors)} failures")


class UAIAuthenticationError(UAIError):
    """Raised when API key, token, or credential validation fails."""


class UAINetworkError(UAIError):
    """Raised on network-level failures (timeouts, connection errors)."""


class UAIRateLimitError(UAIError):
    """Raised when a provider returns a 429 (rate-limited)."""

    def __init__(
        self,
        message: str = "",
        *,
        retry_after: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class ResponseParsingError(UAIError):
    """Raised when a provider response cannot be parsed or validated."""


class FeatureNotSupportedError(UAIError):
    """Raised when a requested feature is not supported by the provider/model."""

    def __init__(
        self,
        message: str = "",
        *,
        feature: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.feature = feature
        full_msg = message or f"Feature '{feature}' is not supported"
        if provider:
            full_msg += f" by provider '{provider}'"
        if model:
            full_msg += f" (model '{model}')"
        super().__init__(full_msg, provider=provider, model=model)


class UAITimeoutError(UAIError):
    """Raised when a request exceeds its configured timeout."""
