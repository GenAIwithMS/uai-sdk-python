"""
Configuration file loader for the Universal AI Provider SDK.

This module supports loading provider configurations from external YAML or
JSON files, merging them with the hardcoded registry defaults.  File
values take precedence, enabling users to customise endpoints, timeouts,
rate limits, or add entirely new providers without touching the SDK source.

Config-file precedence *inside the SDK*:

    User config file  >  Hardcoded defaults (PROVIDER_REGISTRY)
    Environment overrides (env.py)  >  Config file  >  Hardcoded defaults

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from uai.exceptions import ConfigError

from .providers import PROVIDER_REGISTRY, register_provider
from .schema import ProviderConfig

# ---------------------------------------------------------------------------
# Optional YAML support — fall back to JSON-only if PyYAML is absent.
# ---------------------------------------------------------------------------

try:
    import yaml

    _HAS_YAML: bool = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_config_cache: dict[str, ProviderConfig] | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Search paths checked when auto-discovering the config file.
_DEFAULT_CONFIG_PATHS: list[str] = [
    "~/.config/uai/providers.yaml",
    "~/.config/uai/providers.yml",
    "~/.config/uai/providers.json",
    "./providers.yaml",
    "./providers.yml",
    "./providers.json",
]

#: File suffixes recognised as YAML.
_YAML_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")

#: File suffixes recognised as JSON.
_JSON_SUFFIXES: tuple[str, ...] = (".json",)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_config_file() -> Path | None:
    """
    Locate the user configuration file.

    Search order:

    1. `` $UAI_CONFIG_PATH`` environment variable (must point to an existing file).
    2. Each path in :data:`_DEFAULT_CONFIG_PATHS` (user-level then project-level).

    :return: The first existing config file path, or ``None`` if no file is found.
    :raises ConfigError: If ``UAI_CONFIG_PATH`` is set but the file doesn't exist.
    """
    env_path = os.environ.get("UAI_CONFIG_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise ConfigError(
                f"UAI_CONFIG_PATH is set to '{env_path}' but the file does not exist: {path}"
            )
        if path.suffix.lower() not in _YAML_SUFFIXES + _JSON_SUFFIXES:
            raise ConfigError(
                f"UAI_CONFIG_PATH '{env_path}' has an unsupported extension. "
                f"Supported: .yaml, .yml, .json"
            )
        return path

    for path_str in _DEFAULT_CONFIG_PATHS:
        path = Path(path_str).expanduser()
        if path.exists() and path.is_file():
            return path

    return None


# ---------------------------------------------------------------------------
# Raw file loading
# ---------------------------------------------------------------------------


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its parsed contents as a dict."""
    if not _HAS_YAML:
        raise ConfigError(
            "PyYAML is required to load YAML configuration files. "
            "Install the extra:  pip install uai-sdk[yaml]  "
            "or use a .json config file instead."
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        snippet = _format_yaml_error(exc, path)
        raise ConfigError(f"YAML parsing error in {path}:\n{snippet}") from exc
    return data or {}


def _load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file and return its parsed contents as a dict."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"JSON parsing error in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return data or {}


def _load_raw_config(path: Path) -> dict[str, Any]:
    """Dispatch to the correct loader based on file extension."""
    suffix = path.suffix.lower()

    if suffix in _YAML_SUFFIXES:
        return _load_yaml_file(path)
    elif suffix in _JSON_SUFFIXES:
        return _load_json_file(path)
    else:
        raise ConfigError(
            f"Unsupported configuration file format: '{suffix}'. "
            f"Supported formats: .yaml, .yml, .json"
        )


def _format_yaml_error(exc: yaml.YAMLError, path: Path) -> str:
    """Return a human-readable snippet for a YAML parsing error."""
    if hasattr(exc, "problem_mark"):
        mark = exc.problem_mark
        lines = str(path).splitlines()
        line_no = mark.line + 1  # 0-based -> 1-based
        if 0 < line_no <= len(lines):
            snippet = lines[line_no - 1] if lines else "?"
        else:
            snippet = "<unknown>"
        return (
            f"  line {mark.line + 1}, column {mark.column + 1}\n"
            f"  {snippet}\n"
            f"  {' ' * mark.column}^ {exc.problem}"
        )
    return str(exc)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _parse_providers_section(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract and validate the ``providers`` key from raw config dict."""
    if "providers" not in raw:
        raise ConfigError(
            "Configuration file must contain a top-level 'providers' key "
            "mapping provider names to their configs."
        )

    providers = raw["providers"]
    if not isinstance(providers, dict):
        raise ConfigError(f"'providers' must be a mapping (dict), got {type(providers).__name__}.")

    return providers


def _merge_with_defaults(name: str, config_dict: dict[str, Any]) -> ProviderConfig:
    """
    Merge a user-supplied config dict with the hardcoded provider.

    If *name* already exists in :data:`PROVIDER_REGISTRY`, the user's
    values take precedence via a recursive deep-merge (so partial
    overrides of nested dicts like ``regions`` are preserved).
    Otherwise, the dict is treated as a *new* provider definition and
    validated from scratch.

    The registry key (``name``) is authoritative — any ``name`` field
    inside *config_dict* is overwritten to match it.
    """
    if name not in PROVIDER_REGISTRY:
        # New provider — ensure the name matches the YAML/JSON key.
        config_dict = {**config_dict, "name": name}
        return ProviderConfig(**config_dict)

    base = PROVIDER_REGISTRY[name]
    base_dict = base.model_dump()
    # Recursive deep-merge: user dict overrides base dict, with nested
    # dicts merged field-by-field (e.g. adding a new region keeps
    # existing regions intact).
    merged = _deep_merge_dicts(base_dict, config_dict)
    # Enforce that the name matches the registry key.
    merged["name"] = name
    return ProviderConfig(**merged)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge *override* into *base*.

    Nested ``dict`` values are merged recursively; all other types
    (lists, scalars, None) are replaced outright by the override value.
    """
    result: dict[str, Any] = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config_file(path: str | Path) -> dict[str, ProviderConfig]:
    """
    Load and validate provider configs from a single file.

    :param path: Path to a YAML or JSON configuration file.
    :return:  Dict mapping provider name → validated :class:`ProviderConfig`.
    :raises ConfigError: On parse errors, missing ``providers`` key, or
                         schema validation failures.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise ConfigError(f"Configuration file not found: {file_path}")
    if not file_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {file_path}")

    raw = _load_raw_config(file_path)
    providers_raw = _parse_providers_section(raw)

    result: dict[str, ProviderConfig] = {}
    for name, config_dict in providers_raw.items():
        if not isinstance(config_dict, dict):
            raise ConfigError(
                f"Provider '{name}' configuration must be a mapping/dict, "
                f"got {type(config_dict).__name__}."
            )
        try:
            result[name] = _merge_with_defaults(name, config_dict)
        except Exception as exc:
            # Re-raise Pydantic ValidationError with context.
            if not isinstance(exc, ConfigError):
                raise ConfigError(f"Validation failed for provider '{name}': {exc}") from exc
            raise

    return result


def load_config(path: str | Path | None = None) -> dict[str, ProviderConfig]:
    """
    Discover and load the config file, with caching.

    If *path* is ``None``, the file is auto-discovered via
    ``UAI_CONFIG_PATH`` or :data:`_DEFAULT_CONFIG_PATHS`.
    Results are cached; call :func:`clear_cache` to force a reload.

    :return: Dict of validated provider configs (empty dict if no file found).
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    if path is not None:
        _config_cache = load_config_file(path)
    else:
        file_path = find_config_file()
        # No config file on disk — rely entirely on hardcoded registry.
        _config_cache = {} if file_path is None else load_config_file(file_path)

    return _config_cache


def get_config() -> dict[str, ProviderConfig]:
    """Return the cached config, auto-loading if necessary."""
    return load_config()


def clear_cache() -> None:
    """Clear the cached config so the next ``load_config()`` re-reads the file."""
    global _config_cache
    _config_cache = None


def apply_to_registry(
    configs: dict[str, ProviderConfig],
) -> list[str]:
    """
    Register (or override) providers in the global :data:`PROVIDER_REGISTRY`.

    This is called automatically during :class:`~uai.UniversalAI`
    initialisation after :func:`load_config` succeeds.

    :param configs: Dict of validated provider configs from
                    :func:`load_config` or :func:`load_config_file`.
    :return: List of provider names that were added or overridden.
    """
    applied: list[str] = []
    for name, config in configs.items():
        register_provider(config, override=True)
        applied.append(name)
    return applied
