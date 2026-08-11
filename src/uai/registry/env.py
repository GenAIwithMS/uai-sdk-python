"""
Environment variable overrides for provider configurations.

This module allows DevOps and runtime environments to customise provider
settings — endpoints, timeouts, rate limits, credentials — via environment
variables, without modifying the SDK source code or config files.

Override precedence (highest to lowest):

    Environment variables  >  Config file  >  Hardcoded registry

Supported environment variables per provider ``NAME``:

    UAI_PROVIDER_{NAME}_BASE_URL          override the API base URL
    UAI_PROVIDER_{NAME}_TIMEOUT           override request timeout (seconds)
    UAI_PROVIDER_{NAME}_MAX_RETRIES       override max retry count
    UAI_PROVIDER_{NAME}_RATE_LIMIT_RPM    override requests-per-minute limit
    UAI_PROVIDER_{NAME}_RATE_LIMIT_TPM    override tokens-per-minute limit
    UAI_PROVIDER_{NAME}_AUTH_TYPE         override auth method
    UAI_PROVIDER_{NAME}_API_KEY_ENV       override the env var name holding the API key
    UAI_PROVIDER_{NAME}_DEFAULT_MODEL     override the default chat model
    UAI_PROVIDER_{NAME}_DEFAULT_EMBEDDING_MODEL   override the default embeddings model
    UAI_PROVIDER_{NAME}_DEFAULT_RERANK_MODEL      override the default rerank model

Boolean environment variables (for feature flags):

    UAI_PROVIDER_{NAME}_ALLOW_UNKNOWN_MODELS
        Set to false to reject model ids absent from the registry instead of
        passing them through to the provider.

    UAI_PROVIDER_{NAME}_DISABLE_{CAPABILITY}
        Set to true/false to force-disable a capability (e.g. VISION, TOOLS)
        across all models of a provider.  Useful for cost-control or
        compliance overrides.

Note that these are read from the *process environment*.  The SDK does not
parse ``.env`` files itself — load one with ``python-dotenv`` (available as
the ``uai-sdk[dotenv]`` extra) before constructing a client, or export the
variables in your shell.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic import ValidationError

from .providers import PROVIDER_REGISTRY
from .schema import AuthType, ProviderCapabilities, ProviderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping: env-suffix -> (config-field-name, type-converter)
_NUMERIC_FIELD_MAP: dict[str, tuple[str, type]] = {
    "TIMEOUT": ("timeout", float),
    "MAX_RETRIES": ("max_retries", int),
    "RATE_LIMIT_RPM": ("rate_limit_rpm", int),
    "RATE_LIMIT_TPM": ("rate_limit_tpm", int),
}

_STRING_FIELD_MAP: dict[str, tuple[str, type]] = {
    "BASE_URL": ("base_url", str),
    "API_VERSION": ("api_version", str),
    "DOCUMENTATION_URL": ("documentation_url", str),
    "API_KEY_ENV": ("api_key_env_var", str),
    "DEFAULT_MODEL": ("default_model", str),
    "DEFAULT_EMBEDDING_MODEL": ("default_embedding_model", str),
    "DEFAULT_RERANK_MODEL": ("default_rerank_model", str),
}

# Env-suffix -> config-field for plain boolean flags (distinct from the
# DISABLE_{CAPABILITY} pattern, which targets nested model capabilities).
_BOOL_FIELD_MAP: dict[str, str] = {
    "ALLOW_UNKNOWN_MODELS": "allow_unknown_models",
}

# Valid capability field names that can be targeted by feature flags.
_CAPABILITY_FIELDS: list[str] = list(ProviderCapabilities.model_fields.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upper_name(provider_name: str) -> str:
    """Normalise *provider_name* to uppercase for env-var construction."""
    return provider_name.strip().upper()


def _parse_bool(value: str) -> bool:
    """
    Parse a string as a boolean.

    Accepts (case-insensitive): true/false, 1/0, yes/no, on/off, "".
    :raises ValueError: For unrecognised values.
    """
    lower = value.strip().lower()
    if lower in {"true", "1", "yes", "on"}:
        return True
    if lower in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean from '{value}'")


def _safe_int(raw: str, field_name: str, provider_name: str) -> int | None:
    try:
        return int(raw.strip())
    except (ValueError, TypeError):
        _warn_invalid(field_name, raw, provider_name, "integer")
        return None


def _safe_float(raw: str, field_name: str, provider_name: str) -> float | None:
    try:
        return float(raw.strip())
    except (ValueError, TypeError):
        _warn_invalid(field_name, raw, provider_name, "float")
        return None


def _safe_str(raw: str) -> str:
    return raw.strip()


def _warn_invalid(field: str, value: str, provider: str, expected: str) -> None:
    logger.warning(
        "Invalid value '%s' for UAI_PROVIDER_%s_%s: expected %s. Skipping override.",
        value,
        _upper_name(provider),
        field,
        expected,
    )


def _parse_auth_type(raw: str, provider_name: str) -> AuthType | None:
    cleaned = raw.strip().upper()
    try:
        return AuthType[cleaned]
    except KeyError:
        available = ", ".join(a.name for a in AuthType)
        _warn_invalid("AUTH_TYPE", raw, provider_name, f"one of [{available}]")
        return None


# ---------------------------------------------------------------------------
# Per-provider override extraction
# ---------------------------------------------------------------------------


def get_env_overrides(provider_name: str) -> dict[str, Any]:
    """
    Scan the environment and return all overrides for a single provider.

    :param provider_name: Canonical (lowercase) provider name.
    :return: Dict of field-name → parsed-value pairs suitable for
             ``ProviderConfig.model_copy(update=...)``.
    """
    upper = _upper_name(provider_name)
    overrides: dict[str, Any] = {}

    # --- Numeric fields ------------------------------------------------
    for env_suffix, (field, _) in _NUMERIC_FIELD_MAP.items():
        raw = os.environ.get(f"UAI_PROVIDER_{upper}_{env_suffix}")
        if raw is None or not raw.strip():
            continue

        if field == "timeout":
            val = _safe_float(raw, env_suffix, provider_name)
        else:
            val = _safe_int(raw, env_suffix, provider_name)
        if val is not None:
            overrides[field] = val

    # --- String fields -------------------------------------------------
    for env_suffix, (field, _) in _STRING_FIELD_MAP.items():
        raw = os.environ.get(f"UAI_PROVIDER_{upper}_{env_suffix}")
        if raw is None or not raw.strip():
            continue
        overrides[field] = _safe_str(raw)

    # --- Plain boolean fields ------------------------------------------
    for env_suffix, field in _BOOL_FIELD_MAP.items():
        raw = os.environ.get(f"UAI_PROVIDER_{upper}_{env_suffix}")
        if raw is None or not raw.strip():
            continue
        try:
            overrides[field] = _parse_bool(raw)
        except ValueError:
            _warn_invalid(env_suffix, raw, provider_name, "boolean")

    # --- Auth type -----------------------------------------------------
    raw_auth = os.environ.get(f"UAI_PROVIDER_{upper}_AUTH_TYPE")
    if raw_auth is not None and raw_auth.strip():
        parsed = _parse_auth_type(raw_auth, provider_name)
        if parsed is not None:
            overrides["auth_type"] = parsed

    # --- Feature-flag booleans ----------------------------------------
    # Format: UAI_PROVIDER_{NAME}_DISABLE_{CAPABILITY}
    feature_pattern = re.compile(rf"^UAI_PROVIDER_{upper}_DISABLE_([A-Z_]+)$")
    for env_var, raw_val in os.environ.items():
        match = feature_pattern.match(env_var)
        if not match:
            continue
        capability = match.group(1).lower()
        if capability not in _CAPABILITY_FIELDS:
            _warn_invalid(
                f"DISABLE_{match.group(1)}",
                raw_val,
                provider_name,
                f"one of [{', '.join(_CAPABILITY_FIELDS)}]",
            )
            continue
        try:
            flag = _parse_bool(raw_val)
        except ValueError:
            _warn_invalid(f"DISABLE_{match.group(1)}", raw_val, provider_name, "boolean")
            continue
        if flag:
            overrides.setdefault("_disable_capabilities", []).append(capability)

    return overrides


def apply_env_overrides_to_config(
    config: ProviderConfig,
) -> ProviderConfig:
    """
    Return a copy of *config* with environment-variable overrides applied.

    If no env overrides are present, the original (un-copied) config is
    returned for efficiency.
    """
    overrides = get_env_overrides(config.name)

    # Extract model-level capability overrides (feature flags).
    disable_caps: list[str] = overrides.pop("_disable_capabilities", [])
    if not overrides and not disable_caps:
        return config

    if disable_caps:
        # Rebuild models dict with overridden capabilities
        new_models: dict[str, Any] = {}
        for model_id, model in config.models.items():
            # Rebuild capabilities with specific fields set to False
            cap_dict = model.capabilities.model_dump()
            for cap in disable_caps:
                cap_dict[cap] = False
            from .schema import ProviderCapabilities as PC

            new_models[model_id] = model.model_copy(update={"capabilities": PC(**cap_dict)})
        overrides["models"] = new_models

    if not overrides:
        return config

    # Reconstruct rather than ``model_copy(update=...)``: model_copy skips
    # validators, which would let UAI_PROVIDER_X_DEFAULT_MODEL install a
    # default that fails its capability check and only blow up later, deep in
    # a request. Rebuilding runs the schema again at configuration time.
    merged: dict[str, Any] = {**config.model_dump(), **overrides}
    try:
        return ProviderConfig(**merged)
    except ValidationError as exc:
        logger.warning(
            "[uai] environment overrides for provider '%s' are invalid and were "
            "ignored: %s",
            config.name,
            exc,
        )
        return config


def apply_env_overrides(
    configs: dict[str, ProviderConfig] | None = None,
) -> dict[str, ProviderConfig]:
    """
    Apply environment-variable overrides across all provider configs.

    :param configs: Optional dict of configs to override.  Defaults to
                    a copy of :data:`PROVIDER_REGISTRY`.
    :return: New dict with overrides applied.
    """
    if configs is None:
        configs = dict(PROVIDER_REGISTRY)

    result: dict[str, ProviderConfig] = {}
    for name, config in configs.items():
        result[name] = apply_env_overrides_to_config(config)
    return result
