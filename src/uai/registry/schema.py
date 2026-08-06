"""
Provider metadata schema for the Universal AI Provider SDK.

This module defines the Pydantic models that standardize how provider
configurations, model capabilities, pricing, and authentication are
described across the entire SDK.  These models are the single source of
truth for provider metadata and are validated at import time so that
mis-configuration fails fast.

Reference: Implementation Plan — Sub-module 1.0.1
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AuthType(str, Enum):
    """Supported authentication methods for provider API access."""

    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH = "oauth"


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------

class ProviderCapabilities(BaseModel):
    """
    Boolean capability matrix advertised by a provider/model.

    The SDK never *fakes* an implementation for an unsupported capability;
    if a capability is ``False`` and a request is made for it, the SDK
    raises :class:`~uai.exceptions.FeatureNotSupportedError`.
    """

    model_config = ConfigDict(extra="forbid")

    chat: bool = False
    streaming: bool = False
    tools: bool = False
    vision: bool = False
    embeddings: bool = False
    audio: bool = False
    reasoning: bool = False
    rerank: bool = False
    tts: bool = False
    transcription: bool = False


class ProviderPricing(BaseModel):
    """Per-1K-token pricing for a model (in USD)."""

    model_config = ConfigDict(extra="forbid")

    input_cost_per_1k: float = Field(
        default=0.0, ge=0, description="Cost per 1 000 input tokens (USD)."
    )
    output_cost_per_1k: float = Field(
        default=0.0, ge=0, description="Cost per 1 000 output tokens (USD)."
    )

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        """Return the approximate cost in USD for a given token split."""
        return (
            (input_tokens / 1_000) * self.input_cost_per_1k
            + (output_tokens / 1_000) * self.output_cost_per_1k
        )


class ProviderModel(BaseModel):
    """Metadata for a single model offered by a provider."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Machine-readable model identifier.")
    display_name: str = Field(min_length=1, description="Human-readable name.")
    context_window: int = Field(
        gt=0, description="Maximum context window size in tokens."
    )
    max_output_tokens: int = Field(
        gt=0, description="Maximum output tokens per request."
    )
    pricing: ProviderPricing = Field(
        default_factory=ProviderPricing, description="Pricing metadata."
    )
    capabilities: ProviderCapabilities = Field(
        default_factory=ProviderCapabilities,
        description="Capability matrix for this specific model.",
    )
    aliases: List[str] = Field(
        default_factory=list, description="Alternative names for the model."
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model id must not be empty or whitespace.")
        return v.strip()

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Display name must not be empty or whitespace.")
        return v.strip()


class RegionConfig(BaseModel):
    """Override configuration for a specific geographic region."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1, description="Region-specific base URL.")
    auth_type: Optional[AuthType] = Field(
        default=None, description="Override auth type for this region (falls back to provider default)."
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Override environment variable name for the API key in this region.",
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Base URL must not be empty.")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with 'http://' or 'https://'.")
        return cleaned


# ---------------------------------------------------------------------------
# Top-level provider config
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    """
    Complete configuration for a single LLM provider.

    This is the central schema consumed by the registry and all provider
    adapters.  It captures endpoints, authentication, the full model list,
    rate limits, and region overrides.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        description="Canonical provider name (lowercase, no spaces, used as key).",
    )
    display_name: str = Field(
        min_length=1, description="Human-friendly provider name."
    )
    base_url: str = Field(
        min_length=1, description="Base URL for the provider API."
    )
    auth_type: AuthType = Field(description="Authentication method.")
    api_key_env_var: str = Field(
        min_length=1,
        description="Environment variable name from which to read the API key/token.",
    )
    models: Dict[str, ProviderModel] = Field(
        default_factory=dict,
        description="Mapping of model_id -> ProviderModel metadata.",
    )
    default_model: str = Field(
        min_length=1, description="Model used when no model is explicitly requested."
    )
    api_version: str = Field(
        default="v1", description="Provider API version."
    )
    timeout: float = Field(
        default=30.0, gt=0, le=300, description="Default request timeout (seconds)."
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts."
    )
    rate_limit_rpm: Optional[int] = Field(
        default=None, ge=0, description="Requests per minute limit."
    )
    rate_limit_tpm: Optional[int] = Field(
        default=None, ge=0, description="Tokens per minute limit."
    )
    documentation_url: Optional[str] = Field(
        default=None, description="URL to provider documentation."
    )
    organization_required: bool = Field(
        default=False,
        description="Whether an org ID is required (e.g. OpenAI-style org header).",
    )
    regions: Dict[str, RegionConfig] = Field(
        default_factory=dict,
        description="Region-specific overrides keyed by region slug.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Provider name must not be empty.")
        if " " in cleaned:
            raise ValueError("Provider name must not contain spaces.")
        return cleaned

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Base URL must not be empty.")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with 'http://' or 'https://'.")
        return cleaned

    @field_validator("api_key_env_var")
    @classmethod
    def _validate_env_var(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("API key environment variable must not be empty.")
        return cleaned

    @model_validator(mode="after")
    def _validate_models_and_default(self) -> "ProviderConfig":
        # default_model must reference an entry in models
        if self.default_model not in self.models:
            raise ValueError(
                f"default_model '{self.default_model}' is not defined in models. "
                f"Available models: {list(self.models.keys())}"
            )

        # all model ids must be unique — dict keys already enforce this,
        # but aliases must not conflict with real ids
        all_ids: set[str] = set()
        for model_id, model in self.models.items():
            if model_id in all_ids:
                raise ValueError(f"Duplicate model id: {model_id}")
            all_ids.add(model_id)
            for alias in model.aliases:
                if alias in all_ids:
                    raise ValueError(
                        f"Alias '{alias}' of model '{model_id}' conflicts "
                        f"with another model id or alias."
                    )
                all_ids.add(alias)

        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> ProviderCapabilities:
        """
        Aggregated capability matrix across all models in this provider.

        A capability is ``True`` if **any** model advertises it.
        """
        agg = ProviderCapabilities()
        for model in self.models.values():
            for field in ProviderCapabilities.model_fields:
                if getattr(model.capabilities, field):
                    setattr(agg, field, True)
        return agg

    def get_model(self, model_id: str) -> ProviderModel:
        """Return the ``ProviderModel`` for *model_id*, falling back to aliases."""
        if model_id in self.models:
            return self.models[model_id]
        for model in self.models.values():
            if model_id in model.aliases:
                return model
        raise ValueError(
            f"Model '{model_id}' not found for provider '{self.name}'. "
            f"Available: {list(self.models.keys())}"
        )

    @property
    def all_model_ids(self) -> List[str]:
        """Return all primary model ids plus their aliases."""
        ids: list[str] = []
        for model_id, model in self.models.items():
            ids.append(model_id)
            ids.extend(model.aliases)
        return ids
