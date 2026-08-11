"""
Exception hierarchy for the Universal AI Provider SDK.

All SDK-specific exceptions derive from ``UAIError``, enabling
applications to catch any SDK-related failure with a single
``except UAIError`` block.
"""

from __future__ import annotations

from typing import Any


class UAIError(Exception):
    """Base exception for all SDK errors."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        response_body: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class UAIErrorGroup(UAIError):
    """Raised when multiple errors occur (e.g. fallback across providers)."""

    def __init__(self, errors: list[UAIError], message: str = "") -> None:
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
        retry_after: float | None = None,
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
        feature: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        supported_features: list[str] | None = None,
    ) -> None:
        self.feature = feature
        self.supported_features = supported_features
        full_msg = message or f"Feature '{feature}' is not supported"
        if provider:
            full_msg += f" by provider '{provider}'"
        if model:
            full_msg += f" (model '{model}')"
        if supported_features:
            full_msg += f". Supported features: {', '.join(supported_features)}"
        super().__init__(full_msg, provider=provider, model=model)


class ModelNotFoundError(UAIError):
    """
    Raised when a model id cannot be resolved for a provider.

    Distinguishes the two ways a model lookup fails, because the remedies
    differ:

    * the id is unknown to the provider's registry entry — add it via a
      ``providers.yaml`` config file, or allow pass-through with
      ``strict_models=False`` / ``UAI_PROVIDER_{NAME}_ALLOW_UNKNOWN_MODELS``;
    * the id belongs to a *different* registered provider — pass the matching
      ``provider=``.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        available: list[str] | None = None,
        known_from: str | None = None,
    ) -> None:
        self.available = available or []
        self.known_from = known_from

        msg = f"Model '{model}' is not registered for provider '{provider}'."
        if known_from:
            msg += (
                f" It belongs to provider '{known_from}' — "
                f"pass provider='{known_from}' (or drop the provider argument "
                f"to let the SDK infer it)."
            )
        if self.available:
            msg += f" Known models for '{provider}': {', '.join(self.available)}."
        if not known_from:
            msg += (
                " If the provider has released this model since this SDK version, "
                "pass strict_models=False, set "
                f"UAI_PROVIDER_{provider.upper()}_ALLOW_UNKNOWN_MODELS=true, "
                "or declare it in a providers.yaml config file."
            )
        super().__init__(msg, provider=provider, model=model)


class UAITimeoutError(UAIError):
    """Raised when a request exceeds its configured timeout."""


class UAICircuitOpenError(UAIError):
    """
    Raised when a circuit breaker rejects a request while its circuit is open.

    Indicates the provider has sustained repeated failures and is being
    fast-failed until the breaker's reset timeout elapses (Module 1.4.2).
    """


class ConfigError(UAIError):
    """Raised when provider configuration cannot be loaded or validated."""
