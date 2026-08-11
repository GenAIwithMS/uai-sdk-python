"""
Provider metadata schema for the Universal AI Provider SDK.

This module defines the Pydantic models that standardize how provider
configurations, model capabilities, pricing, and authentication are
described across the entire SDK.  These models are the single source of
truth for provider metadata and are validated at import time so that
mis-configuration fails fast.

"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

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
        return (input_tokens / 1_000) * self.input_cost_per_1k + (
            output_tokens / 1_000
        ) * self.output_cost_per_1k


class ProviderModel(BaseModel):
    """Metadata for a single model offered by a provider."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Machine-readable model identifier.")
    display_name: str = Field(min_length=1, description="Human-readable name.")
    context_window: int = Field(gt=0, description="Maximum context window size in tokens.")
    max_output_tokens: int = Field(gt=0, description="Maximum output tokens per request.")
    pricing: ProviderPricing = Field(
        default_factory=ProviderPricing, description="Pricing metadata."
    )
    capabilities: ProviderCapabilities = Field(
        default_factory=ProviderCapabilities,
        description="Capability matrix for this specific model.",
    )
    aliases: list[str] = Field(default_factory=list, description="Alternative names for the model.")

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
    auth_type: AuthType | None = Field(
        default=None,
        description="Override auth type for this region (falls back to provider default).",
    )
    api_key_env_var: str | None = Field(
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
    display_name: str = Field(min_length=1, description="Human-friendly provider name.")
    base_url: str = Field(min_length=1, description="Base URL for the provider API.")
    auth_type: AuthType = Field(description="Authentication method.")
    api_key_env_var: str = Field(
        min_length=1,
        description="Environment variable name from which to read the API key/token.",
    )
    models: dict[str, ProviderModel] = Field(
        default_factory=dict,
        description="Mapping of model_id -> ProviderModel metadata.",
    )
    default_model: str = Field(
        min_length=1, description="Chat model used when no model is explicitly requested."
    )
    default_embedding_model: str | None = Field(
        default=None,
        description="Model used by ``embed()`` when none is requested. Falls back to the "
        "first model advertising the ``embeddings`` capability.",
    )
    default_rerank_model: str | None = Field(
        default=None,
        description="Model used by ``rerank()`` when none is requested. Falls back to the "
        "first model advertising the ``rerank`` capability.",
    )
    allow_unknown_models: bool = Field(
        default=True,
        description="When True, a model id absent from ``models`` is passed through to the "
        "provider with permissive capabilities instead of raising. Keeps the SDK usable on "
        "the day a provider ships a new model, at the cost of pre-flight capability checks "
        "for that id. Set False (or ``strict_models=True`` on the client) to hard-fail.",
    )
    api_version: str = Field(default="v1", description="Provider API version.")
    timeout: float = Field(
        default=30.0, gt=0, le=300, description="Default request timeout (seconds)."
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts.")
    rate_limit_rpm: int | None = Field(default=None, ge=0, description="Requests per minute limit.")
    rate_limit_tpm: int | None = Field(default=None, ge=0, description="Tokens per minute limit.")
    documentation_url: str | None = Field(
        default=None, description="URL to provider documentation."
    )
    organization_required: bool = Field(
        default=False,
        description="Whether an org ID is required (e.g. OpenAI-style org header).",
    )
    regions: dict[str, RegionConfig] = Field(
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
    def _validate_models_and_default(self) -> ProviderConfig:
        # Each default must resolve to a real entry (id *or* alias).  Aliases
        # are accepted so a config file can point a default at, say,
        # "kimi-latest" without restating the canonical id.
        for field_name, capability in (
            ("default_model", "chat"),
            ("default_embedding_model", "embeddings"),
            ("default_rerank_model", "rerank"),
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            found = self._lookup(value)
            if found is None:
                # An unregistered default is legitimate when pass-through is
                # on: it is how a user points at a model the provider shipped
                # after this SDK release.  Under strict_models it is a typo.
                if self.allow_unknown_models:
                    logger.warning(
                        "[uai] provider '%s': %s is '%s', which is not among its "
                        "declared models (%s). It will be sent to the provider "
                        "as-is; set allow_unknown_models=false to make this an error.",
                        self.name,
                        field_name,
                        value,
                        ", ".join(self.models) or "none",
                    )
                    continue
                raise ValueError(
                    f"{field_name} '{value}' is not defined in models. "
                    f"Available models: {list(self.models.keys())}"
                )
            model = found[1]
            declared = model.capabilities.model_dump()
            if not any(declared.values()):
                # A model that declares no capabilities at all is
                # under-specified, not wrong — common in hand-written config
                # files and minimal fixtures. Only contradict an explicit
                # declaration, never an absent one.
                continue
            if not declared.get(capability):
                raise ValueError(
                    f"{field_name} '{value}' does not advertise the '{capability}' "
                    f"capability. Pick a model whose capabilities include it."
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
        """
        Return the ``ProviderModel`` for *model_id*, falling back to aliases.

        Strict: unknown ids always raise.  Use :meth:`resolve_model` for the
        pass-through behaviour that honours ``allow_unknown_models``.
        """
        resolved = self._lookup(model_id)
        if resolved is None:
            raise ValueError(
                f"Model '{model_id}' not found for provider '{self.name}'. "
                f"Available: {list(self.models.keys())}"
            )
        return resolved[1]

    def _lookup(self, model_id: str) -> tuple[str, ProviderModel] | None:
        """Resolve *model_id* (id or alias) to ``(canonical_id, model)``."""
        direct = self.models.get(model_id)
        if direct is not None:
            return model_id, direct
        for mid, model in self.models.items():
            if model_id in model.aliases:
                return mid, model
        return None

    def knows_model(self, model_id: str) -> bool:
        """Return True if *model_id* is a known id or alias for this provider."""
        return self._lookup(model_id) is not None

    def resolve_model(
        self,
        model_id: str,
        *,
        allow_unknown: bool | None = None,
    ) -> tuple[str, ProviderModel, bool]:
        """
        Resolve *model_id* to ``(canonical_id, model, is_unregistered)``.

        Aliases are dereferenced to their canonical id.  When *model_id* is
        unknown and pass-through is permitted, a permissive placeholder is
        synthesized so a model released after this SDK version still reaches
        the provider — the third tuple element flags that case so callers can
        soften capability enforcement and warn.

        :param allow_unknown: Overrides ``self.allow_unknown_models`` when set.
        :raises ValueError: If the model is unknown and pass-through is off.
        """
        found = self._lookup(model_id)
        if found is not None:
            return found[0], found[1], False

        permitted = self.allow_unknown_models if allow_unknown is None else allow_unknown
        if not permitted:
            raise ValueError(
                f"Model '{model_id}' not found for provider '{self.name}'. "
                f"Available: {list(self.models.keys())}"
            )
        return model_id, self._synthesize_model(model_id), True

    def _synthesize_model(self, model_id: str) -> ProviderModel:
        """
        Build a permissive :class:`ProviderModel` for an unregistered id.

        Capabilities are the union advertised by this provider's known models
        rather than blanket ``True``: it keeps genuinely impossible requests
        (asking a rerank-less provider to rerank) failing fast, while never
        blocking a request the provider can plausibly serve.  Context and
        pricing are unknown, so context/output limits take the provider's
        maximum and pricing stays zero — callers must not treat cost estimates
        for an unregistered model as authoritative.
        """
        known = list(self.models.values())
        return ProviderModel(
            id=model_id,
            display_name=f"{self.display_name} {model_id} (unregistered)",
            context_window=max((m.context_window for m in known), default=128_000),
            max_output_tokens=max((m.max_output_tokens for m in known), default=4_096),
            capabilities=self.capabilities,
        )

    def default_model_for(self, capability: str) -> str:
        """
        Return the default model id for *capability* (``chat``/``embeddings``/``rerank``).

        Resolution order: the explicitly configured per-modality default, then
        the first registered model advertising the capability, then
        ``default_model``.  Without this, ``embed()`` would inherit the *chat*
        default and fail a capability check on a provider that does offer
        embeddings.
        """
        explicit = {
            "chat": self.default_model,
            "embeddings": self.default_embedding_model,
            "rerank": self.default_rerank_model,
        }.get(capability)
        if explicit:
            return explicit
        for model_id, model in self.models.items():
            if getattr(model.capabilities, capability, False):
                return model_id
        return self.default_model

    @property
    def all_model_ids(self) -> list[str]:
        """Return all primary model ids plus their aliases."""
        ids: list[str] = []
        for model_id, model in self.models.items():
            ids.append(model_id)
            ids.extend(model.aliases)
        return ids
