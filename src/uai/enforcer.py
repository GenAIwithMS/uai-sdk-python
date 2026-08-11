"""
Capability Matrix Enforcer (Module 1.3.1).

When a developer invokes a method on the :class:`~uai.UniversalAI` client,
the client interrogates the active provider's capability matrix.  If an
unsupported feature is requested — e.g. passing an image payload to the
text-only DeepSeek adapter — the enforcer instantly halts execution and
raises :class:`~uai.exceptions.FeatureNotSupportedError` *before* any
network egress.

This explicit, immediate failure mechanism:

* prevents malformed requests from consuming network bandwidth,
* limits unnecessary exposure of API keys, and
* guarantees the SDK never silently ignores developer intent.

The enforcer merges two sources of truth:

1. the **registry model capabilities** (per-model ``ProviderCapabilities``),
   and
2. the **adapter's ``capabilities()`` matrix** (per-provider, the
   authoritative statement of what the adapter actually implements).

A feature is supported only if *both* sources report ``True``.  This
ensures the SDK never fakes an implementation that the active adapter does
not provide.
"""

from __future__ import annotations

from uai.adapters.base_adapter import BaseProviderAdapter
from uai.exceptions import FeatureNotSupportedError
from uai.registry import get_provider_config
from uai.registry.schema import ProviderCapabilities, ProviderConfig, ProviderModel

_VALID_CAPABILITIES: tuple[str, ...] = tuple(ProviderCapabilities.model_fields.keys())


class CapabilityMatrixEnforcer:
    """
    Enforce the capability matrix for a specific provider/model pair.

    Instances are cheap to create and hold no network state, so the client
    builds one per request at the public-method boundary.

    Example:
        >>> enforcer = CapabilityMatrixEnforcer("deepseek", "deepseek-chat")
        >>> enforcer.supports("chat")
        True
        >>> enforcer.assert_supported("vision")  # raises FeatureNotSupportedError
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        adapter: BaseProviderAdapter | None = None,
        config: ProviderConfig | None = None,
        model_info: ProviderModel | None = None,
    ) -> None:
        """
        Args:
            provider: Canonical provider name (e.g. ``'deepseek'``).
            model: Model id or alias (e.g. ``'deepseek-v4-flash'``).
            adapter: Optional adapter instance whose ``capabilities()``
                matrix is cross-checked against the registry.
            config: Optional pre-resolved :class:`ProviderConfig`.  When
                omitted, the provider is looked up in the registry.
            model_info: Optional pre-resolved :class:`ProviderModel`.  The
                client passes this so an unregistered-but-permitted model
                keeps the placeholder metadata it was resolved with, rather
                than being looked up a second time and rejected.

        :raises ValueError: If the provider or model is unknown.
        """
        provider_lower = provider.lower()
        self.provider = provider_lower
        self._config = config or get_provider_config(provider_lower)
        if model_info is not None:
            self._model_id, self._model_info = model, model_info
        else:
            self._model_id, self._model_info = self._resolve_model(self._config, model)
        self._adapter = adapter
        # Cache the adapter matrix once — it is read on every supports() call.
        self._adapter_matrix: dict[str, bool] | None = (
            adapter.capabilities() if adapter is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """The resolved canonical model id (aliases are dereferenced)."""
        return self._model_id

    @property
    def model_info(self) -> ProviderModel:
        """The registry metadata for the resolved model."""
        return self._model_info

    def supports(self, feature: str) -> bool:
        """
        Return whether *feature* is supported for this provider/model.

        A feature is supported only when both the registry model
        capabilities and (if an adapter is attached) the adapter's
        ``capabilities()`` matrix report ``True``.

        :param feature: Capability field name (e.g. ``'vision'``).
        :raises ValueError: If *feature* is not a known capability.
        """
        self._validate_feature(feature)
        if not getattr(self._model_info.capabilities, feature):
            return False
        if self._adapter_matrix is not None:
            return bool(self._adapter_matrix.get(feature, False))
        return True

    def assert_supported(
        self,
        feature: str,
        *,
        message: str = "",
    ) -> None:
        """
        Raise :class:`FeatureNotSupportedError` if *feature* is unsupported.

        :param feature: Capability field name (e.g. ``'vision'``).
        :param message: Optional message prefix.  The standard
            ``Feature '...' is not supported ...`` text is used when empty;
            when provided, the provider/model suffix and the supported
            features list are still appended by the exception.
        :raises FeatureNotSupportedError: If the feature is unsupported.
        :raises ValueError: If *feature* is not a known capability.
        """
        if self.supports(feature):
            return
        raise FeatureNotSupportedError(
            message,
            feature=feature,
            provider=self.provider,
            model=self.model,
            supported_features=self.supported_features() or None,
        )

    def supported_features(self) -> list[str]:
        """Return the list of capabilities enabled for this provider/model."""
        return [name for name in _VALID_CAPABILITIES if self.supports(name)]

    def unsupported_features(self) -> list[str]:
        """Return the list of capabilities *not* enabled for this provider/model."""
        return [name for name in _VALID_CAPABILITIES if not self.supports(name)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_model(
        self,
        config: ProviderConfig,
        model_id: str,
    ) -> tuple[str, ProviderModel]:
        """Resolve *model_id* (or an alias) to its canonical id and metadata."""
        model_info = config.models.get(model_id)
        resolved_id = model_id

        if model_info is None:
            for mid, info in config.models.items():
                if model_id in info.aliases:
                    model_info = info
                    resolved_id = mid
                    break

        if model_info is None:
            raise ValueError(
                f"Model '{model_id}' not found for provider '{self.provider}'. "
                f"Available: {', '.join(config.models.keys())}"
            )
        return resolved_id, model_info

    @staticmethod
    def _validate_feature(feature: str) -> None:
        if feature not in _VALID_CAPABILITIES:
            raise ValueError(
                f"Unknown capability '{feature}'. "
                f"Valid capabilities: {', '.join(_VALID_CAPABILITIES)}"
            )


__all__ = ["CapabilityMatrixEnforcer"]
