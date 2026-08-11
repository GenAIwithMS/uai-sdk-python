"""
UniversalAI client orchestrator for the Universal AI Provider SDK.

This is the primary entry point for the SDK. It handles:
- Client instantiation with provider and credential configuration
- UnifiedRequest construction from developer inputs
- Capability enforcement
- Provider adapter delegation
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, cast

import httpx

from uai.adapters.base_adapter import BaseProviderAdapter
from uai.enforcer import CapabilityMatrixEnforcer
from uai.exceptions import (
    FeatureNotSupportedError,
    ModelNotFoundError,
    UAIAuthenticationError,
    UAIError,
    UAINetworkError,
    UAIRateLimitError,
    UAITimeoutError,
)
from uai.middleware.base import BaseMiddleware, MiddlewareContext
from uai.middleware.engine import MiddlewareEngine
from uai.middleware.retry import RetryMiddleware
from uai.models import (
    ChatMessage,
    EmbeddingsResponse,
    FinishReason,
    ImageContent,
    RerankResponse,
    Role,
    StreamChunk,
    ToolDefinition,
    UnifiedRequest,
    UnifiedResponse,
    UsageMetrics,
)
from uai.registry import apply_env_overrides, get_provider_config, load_config
from uai.registry.providers import PROVIDER_REGISTRY, find_providers_for_model
from uai.registry.schema import ProviderConfig, ProviderModel
from uai.structured import build_schema_prompt, parse_structured_output

logger = logging.getLogger(__name__)

# Adapter classes are loaded lazily (Module 1.6.1 — resource footprint):
# importing every provider adapter eagerly at ``import uai`` adds ~7 MB of
# marginal memory for users of a single provider. Each spec maps a provider
# name to ``(module, class_name)`` resolved on first use.
_ADAPTER_SPECS: dict[str, tuple[str, str]] = {
    "deepseek": ("uai.adapters.deepseek", "DeepSeekAdapter"),
    "qwen": ("uai.adapters.qwen", "QwenAdapter"),
    "glm": ("uai.adapters.glm", "GLMAdapter"),
    "kimi": ("uai.adapters.kimi", "KimiAdapter"),
    "stepfun": ("uai.adapters.stepfun", "StepFunAdapter"),
    "doubao": ("uai.adapters.doubao", "DoubaoAdapter"),
    "minimax": ("uai.adapters.minimax", "MiniMaxAdapter"),
    "hunyuan": ("uai.adapters.hunyuan", "HunyuanAdapter"),
}


class UniversalAI:
    """
    The primary orchestrator for AI provider interactions.

    This client normalizes requests and routes them to the appropriate
    provider adapter based on configuration or explicit selection.

    Example:
        >>> client = UniversalAI(api_key=DEEPSEEK_API_KEY)
        >>> response = client.chat(messages=[{"role": "user", "content": "Hello"}])
        >>> print(response.content)
    """

    #: Provider assumed when neither ``provider=`` nor an inferable ``model=``
    #: is supplied.
    DEFAULT_PROVIDER = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        credentials: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        strict_models: bool | None = None,
        config_path: str | Path | None = None,
    ):
        """
        Initialize the UniversalAI client.

        Args:
            api_key: API key for *provider*. Scoped to that provider only — a
                per-call ``provider=`` override resolves its own key from the
                environment instead (see :meth:`_get_api_key`).
            provider: Default provider to use (e.g., 'deepseek', 'qwen').
                When omitted, it is inferred from *model* if that model is
                registered to exactly one provider, else defaults to
                :data:`DEFAULT_PROVIDER`.
            model: Default model for this client, in the style of
                ``ChatGroq(model=...)``. An id the registry does not know is
                still accepted and forwarded to the provider verbatim, unless
                ``strict_models=True``. A model registered to a *different*
                provider than the one selected is rejected immediately.
            credentials: Credential dictionary for *provider*. Same scoping as
                ``api_key``.
            timeout: Request timeout in seconds. Takes precedence over the
                registry value and over ``UAI_PROVIDER_{NAME}_TIMEOUT``, and
                applies to every provider this client calls.
            max_retries: When set above zero, enable automatic retries for
                transient failures. This is shorthand for registering a
                :class:`~uai.middleware.retry.RetryMiddleware`, placed inside
                every middleware added via :meth:`use`. Leave unset for no
                retries — retrying stays opt-in.
            strict_models: Force model-id validation on (``True``) or off
                (``False``) for every provider, overriding each provider's
                ``allow_unknown_models``. Leave ``None`` to honour the
                per-provider setting (pass-through by default).
            config_path: Explicit path to a ``providers.yaml``/``.json``
                config file. When omitted the standard search paths and
                ``UAI_CONFIG_PATH`` are used.

        Raises:
            ValueError: If the provider is unknown, if *model* belongs to a
                different provider, or if *model* is unknown while strict
                model validation is in effect.
        """
        self._strict_models = strict_models
        # Precedence: environment > config file > built-in registry.  The
        # loader was previously exported but never invoked, so a providers.yaml
        # on disk had no effect; wiring it here is what makes the documented
        # "add a model without touching the SDK" workflow real.
        merged: dict[str, ProviderConfig] = dict(PROVIDER_REGISTRY)
        merged.update(load_config(config_path))
        self._all_configs = apply_env_overrides(merged)

        provider_lower = self._select_provider(provider, model)
        if provider_lower not in self._all_configs:
            available = ", ".join(self._all_configs.keys())
            raise ValueError(
                f"Provider '{provider}' is not available. Available providers: {available}"
            )

        self._default_provider = provider_lower
        base_config = self._all_configs[provider_lower]

        self._config: ProviderConfig = base_config.model_copy()

        # Held on the client and applied at the point of use rather than
        # written into a ProviderConfig.  ``_config_for`` hands back shared
        # registry entries, so mutating one would leak this client's timeout
        # into every other client in the process, and copying a config per
        # request would cost more than the <5ms overhead budget allows.
        self._timeout = timeout

        # ``max_retries`` is shorthand for a RetryMiddleware.  It is composed
        # innermost (see ``_run_pipeline``) so anything registered through
        # ``use()`` wraps it -- an open circuit breaker then short-circuits
        # without consuming attempts, and a cache hit skips retrying entirely.
        # Retrying stays strictly opt-in: the registry's ``max_retries``
        # default is never enough to switch it on by itself.
        self._auto_retry: RetryMiddleware | None = (
            RetryMiddleware(max_retries=max_retries)
            if max_retries is not None and max_retries > 0
            else None
        )

        # Credentials passed to the constructor belong to ``provider`` and to
        # no other.  They are deliberately *not* used as a global fallback:
        # a per-call ``provider=`` override must never reuse them, or one
        # provider's key would be transmitted to another provider's API.
        # Every other provider resolves its own key at call time.
        if credentials is not None:
            self._credentials = dict(credentials)
        elif api_key is not None:
            self._credentials = {"api_key": api_key}
        else:
            self._credentials = {}

        # Defaults are keyed by provider.  A single ``_default_model`` string
        # was the root of the cross-provider bug: ``chat(provider='qwen')`` on
        # a DeepSeek-defaulted client inherited 'deepseek-chat' and failed
        # lookup against Qwen's catalogue.  A model given here belongs to the
        # provider it was given with, and to no other.
        self._default_models: dict[str, str] = {}
        if model is not None:
            self._validate_constructor_model(provider_lower, model)
            self._default_models[provider_lower] = model

        self._adapters: dict[str, BaseProviderAdapter] = {}
        self._engine = MiddlewareEngine()

        logger.debug(
            f"UniversalAI client initialized with provider={self._default_provider}, "
            f"model={self._model_for(self._default_provider, None, 'chat')}"
        )

    # ------------------------------------------------------------------
    # Provider / model resolution
    # ------------------------------------------------------------------

    @property
    def _default_model(self) -> str:
        """
        The default chat model for this client's default provider.

        Retained as a read-only property for backwards compatibility; internal
        code resolves through :meth:`_model_for` so that per-call
        ``provider=`` overrides pick up *their* provider's default.
        """
        return self._model_for(self._default_provider, None, "chat")

    def _select_provider(self, provider: str | None, model: str | None) -> str:
        """
        Choose the client's default provider, inferring it from *model* if needed.

        Lets ``UniversalAI(model="glm-4.7")`` route without naming a provider,
        mirroring how per-provider LangChain classes make the provider
        implicit. Inference only fires when the model maps to exactly one
        registered provider; ambiguity and unknown ids are reported rather
        than guessed, because guessing would send a key to the wrong vendor.
        """
        if provider is not None:
            return provider.strip().lower()
        if model is None:
            return self.DEFAULT_PROVIDER

        candidates = find_providers_for_model(model, self._all_configs)
        if len(candidates) == 1:
            logger.debug("[uai] inferred provider '%s' from model '%s'", candidates[0], model)
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Model '{model}' is registered to multiple providers "
                f"({', '.join(candidates)}). Pass provider= to disambiguate."
            )
        raise ValueError(
            f"Cannot infer a provider for model '{model}': no registered provider "
            f"declares it. Pass provider= explicitly (e.g. "
            f"UniversalAI(provider='deepseek', model='{model}')). "
            f"Registered providers: {', '.join(self._all_configs)}."
        )

    def _allow_unknown_for(self, config: ProviderConfig) -> bool:
        """Whether unregistered model ids may pass through for *config*."""
        if self._strict_models is not None:
            return not self._strict_models
        return config.allow_unknown_models

    def _validate_constructor_model(self, provider_lower: str, model: str) -> None:
        """
        Fail fast when the constructor's ``model`` cannot serve *provider*.

        Previously any string was accepted here and only surfaced at the first
        ``chat()`` call, far from the mistake. A model belonging to another
        registered provider is always an error — pass-through must not ship a
        Qwen id to DeepSeek's endpoint.
        """
        config = self._all_configs[provider_lower]
        if config.knows_model(model):
            return

        elsewhere = find_providers_for_model(model, self._all_configs)
        if elsewhere:
            raise ModelNotFoundError(
                model,
                provider_lower,
                available=list(config.models),
                known_from=elsewhere[0],
            )
        if not self._allow_unknown_for(config):
            raise ModelNotFoundError(model, provider_lower, available=list(config.models))
        logger.warning(
            "[uai] model '%s' is not in the registry for provider '%s'; forwarding it "
            "as-is. Capability checks for this model fall back to the provider's "
            "aggregate capabilities.",
            model,
            provider_lower,
        )

    def _config_for(self, provider_lower: str) -> ProviderConfig:
        """Return the effective :class:`ProviderConfig` for *provider_lower*."""
        config = self._all_configs.get(provider_lower)
        if config is not None:
            return config
        return get_provider_config(provider_lower)

    def _model_for(
        self,
        provider_lower: str,
        model: str | None,
        capability: str = "chat",
    ) -> str:
        """
        Resolve the model id to use for *provider_lower* and *capability*.

        Order: the explicit per-call model, then this client's default for
        *that* provider, then the provider's own default for the requested
        modality. The last step is why ``embed()`` no longer inherits a chat
        model and fails a capability check on providers that do offer
        embeddings.
        """
        if model:
            return model
        if capability == "chat":
            pinned = self._default_models.get(provider_lower)
            if pinned:
                return pinned
        return self._config_for(provider_lower).default_model_for(capability)

    def _resolve(
        self,
        provider_lower: str,
        model: str | None,
        capability: str = "chat",
    ) -> tuple[ProviderConfig, str, ProviderModel]:
        """
        Resolve provider config, canonical model id, and model metadata.

        Single funnel for what used to be three divergent alias-resolution
        implementations (client, enforcer, schema), each with its own error
        text and its own idea of what counts as unknown.
        """
        config = self._config_for(provider_lower)
        model_id = self._model_for(provider_lower, model, capability)
        try:
            resolved_id, info, unregistered = config.resolve_model(
                model_id, allow_unknown=self._allow_unknown_for(config)
            )
        except ValueError as exc:
            elsewhere = find_providers_for_model(model_id, self._all_configs)
            raise ModelNotFoundError(
                model_id,
                provider_lower,
                available=list(config.models),
                known_from=elsewhere[0] if elsewhere else None,
            ) from exc
        if unregistered:
            logger.debug(
                "[uai] forwarding unregistered model '%s' to provider '%s'",
                resolved_id,
                provider_lower,
            )
        return config, resolved_id, info

    def use(self, middleware: BaseMiddleware | list[BaseMiddleware]) -> UniversalAI:
        """
        Register one or more middleware instances (opt-in pipeline).

        Middleware hooks run in registration order for ``before_request``
        and in reverse order for ``after_response``/``on_error``.

        Registering a :class:`~uai.middleware.retry.RetryMiddleware` here
        supersedes the constructor's ``max_retries`` shorthand. Composing both
        would nest two retry loops and multiply the request count, so the
        explicit middleware wins and the shorthand is dropped.

        Args:
            middleware: A single middleware or a list of middleware.

        Returns:
            The client, for chaining.
        """
        items = middleware if isinstance(middleware, list) else [middleware]
        if self._auto_retry is not None and any(isinstance(m, RetryMiddleware) for m in items):
            logger.warning(
                "[uai] a RetryMiddleware was registered explicitly, so the client's "
                "max_retries=%d is ignored (nesting both would multiply attempts)",
                self._auto_retry.max_retries,
            )
            self._auto_retry = None

        self._engine.use(middleware)
        return self

    def _run_pipeline(
        self,
        operation: str,
        provider: str,
        model: str,
        request: Any,
        execute_fn: Callable[[MiddlewareContext], Any],
    ) -> Any:
        """Run before -> execute -> after around a non-streaming callable (Module 1.4.1)."""
        return self._engine.run(
            operation, provider, model, request, self._with_auto_retry(execute_fn)
        )

    def _run_stream_pipeline(
        self,
        operation: str,
        provider: str,
        model: str,
        request: Any,
        stream_fn: Callable[[MiddlewareContext], Any],
    ) -> Any:
        """Run before -> execute around a streaming callable; after on finish (Module 1.4.1)."""
        return self._engine.run_stream(
            operation, provider, model, request, self._with_auto_retry(stream_fn)
        )

    def _with_auto_retry(
        self, execute_fn: Callable[[MiddlewareContext], Any]
    ) -> Callable[[MiddlewareContext], Any]:
        """
        Wrap *execute_fn* in the constructor's retry policy, if one is set.

        Wrapping the innermost callable — rather than registering the retry in
        the middleware list — keeps it beneath everything added via
        :meth:`use`, which is the topology the middleware are documented to
        expect. ``RetryMiddleware.execute`` dispatches on
        ``context.request.stream`` itself, so this is correct for streaming
        and non-streaming operations alike.
        """
        auto_retry = self._auto_retry
        if auto_retry is None:
            return execute_fn

        def _retrying(context: MiddlewareContext) -> Any:
            return auto_retry.execute(lambda: execute_fn(context), context)

        return _retrying

    def _get_adapter(self, provider_lower: str) -> BaseProviderAdapter:
        """Return the adapter instance for a provider, importing it lazily."""
        if provider_lower not in self._adapters:
            spec = _ADAPTER_SPECS.get(provider_lower)
            if spec is None:
                available = ", ".join(_ADAPTER_SPECS.keys())
                raise ValueError(
                    f"No adapter for provider '{provider_lower}'. Available: {available}"
                )
            module_name, class_name = spec
            adapter_cls = getattr(importlib.import_module(module_name), class_name)
            self._adapters[provider_lower] = adapter_cls()
        return self._adapters[provider_lower]

    def _timeout_for(self, config: ProviderConfig) -> float:
        """
        Resolve the request timeout for *config*'s provider.

        A timeout passed to the constructor takes precedence over the
        registry value and over ``UAI_PROVIDER_{NAME}_TIMEOUT`` (both of which
        reach us through ``config``), matching the documented precedence of
        constructor arguments over environment configuration.
        """
        return self._timeout if self._timeout is not None else config.timeout

    def _get_api_key(self, config: ProviderConfig) -> str | None:
        """
        Resolve the API key for the provider described by *config*.

        Constructor credentials are scoped to the client's default provider.
        Any other provider — reached through a per-call ``provider=``
        override — resolves its own key from its ``api_key_env_var``, so a
        credential is never transmitted to a provider it was not issued for.

        Falling through to the environment on every call also means a rotated
        key is picked up without rebuilding the client.

        Args:
            config: Configuration of the provider being called.

        Returns:
            The API key, or ``None`` if no credential is available.
        """
        if config.name.lower() == self._default_provider:
            api_key = self._credentials.get("api_key")
            if api_key is not None:
                return str(api_key)

        api_key_env = config.api_key_env_var
        result = os.environ.get(api_key_env) if api_key_env else None
        return str(result) if result is not None else None

    def _enforcer(
        self,
        provider_lower: str,
        model: str | None,
        capability: str = "chat",
    ) -> CapabilityMatrixEnforcer:
        """
        Build a capability enforcer for the active provider/model/adapter.

        The enforcer merges the registry model capabilities with the
        adapter's ``capabilities()`` matrix (Module 1.3.1).
        """
        config, resolved_id, info = self._resolve(provider_lower, model, capability)
        adapter = self._get_adapter(provider_lower)
        return CapabilityMatrixEnforcer(
            provider_lower,
            resolved_id,
            adapter=adapter,
            config=config,
            model_info=info,
        )

    def supports(
        self,
        feature: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> bool:
        """
        Pre-flight check: does the resolved provider/model support *feature*?

        Args:
            feature: Capability name (e.g. ``'vision'``, ``'tools'``).
            provider: Provider name (defaults to the client's provider).
            model: Model id or alias. Defaults to the default model of the
                *resolved* provider for the modality *feature* implies.

        Returns:
            ``True`` if the feature is supported by both the registry model
            capabilities and the active adapter's matrix.
        """
        provider_lower = (provider or self._default_provider).lower()
        capability = feature if feature in ("embeddings", "rerank") else "chat"
        return self._enforcer(provider_lower, model, capability).supports(feature)

    @staticmethod
    def _messages_contain_images(messages: list[ChatMessage]) -> bool:
        """Return True if any message carries an ``ImageContent`` block."""
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ImageContent):
                        return True
        return False

    def chat(
        self,
        messages: list[dict[str, Any] | ChatMessage] | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        stream: bool = False,
        stream_callback: Callable[[StreamChunk], None] | None = None,
        tools: list | None = None,
        output_schema: type | None = None,
        **kwargs: Any,
    ) -> UnifiedResponse | Iterator[StreamChunk]:
        """
        Execute a chat completion request.

        Args:
            messages: List of conversation messages.
            provider: Target provider (uses default if not specified).
            model: Target model (uses default if not specified).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 to 2.0).
            top_p: Top-p nucleus sampling parameter (0.0 to 1.0).
            stop: Stop sequence(s) where generation should halt.
            stream: Whether to stream the response.
            stream_callback: Optional callback for streaming chunks.
            tools: Optional list of tool definitions.
            output_schema: Pydantic model for structured output validation.
            **kwargs: Additional provider-specific parameters.

        Returns:
            UnifiedResponse for non-streaming, or iterator of StreamChunk for streaming.
        """
        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]

        # Normalize messages
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized_messages.append(msg)
            elif isinstance(msg, dict):
                normalized_messages.append(ChatMessage(**msg))
            else:
                raise ValueError(f"Message must be ChatMessage or dict, got {type(msg)}")

        # Build unified request
        request = UnifiedRequest(
            provider=provider,
            model=model,
            messages=normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream,
        )

        # Apply tools if provided — normalize dicts to ToolDefinition so the
        # request model stays consistent (the constructor validator does this
        # on construction, but tools are assigned after it).
        if tools is not None:
            request.tools = [ToolDefinition(**t) if isinstance(t, dict) else t for t in tools]
        if output_schema is not None:
            request.output_schema = output_schema

        # Apply additional kwargs.  Unknown names are rejected rather than
        # dropped: silently discarding `seed=42` or a misspelled parameter
        # sends a request that quietly ignores what the caller asked for.
        unknown = [key for key in kwargs if key not in UnifiedRequest.model_fields]
        if unknown:
            raise TypeError(
                f"chat() got unexpected keyword argument(s): {', '.join(sorted(unknown))}. "
                f"Supported request fields: {', '.join(sorted(UnifiedRequest.model_fields))}."
            )
        for key, value in kwargs.items():
            if value is not None:
                setattr(request, key, value)

        operation_provider = (request.provider or self._default_provider).lower()

        # Module 1.3.1 — capability matrix enforcement.  Halt before any
        # middleware or network work if the requested features are unsupported.
        enforcer = self._enforcer(operation_provider, request.model)
        # Label middleware (metrics, cache keys, traces) with the *canonical*
        # id so an alias and its target aggregate as one series instead of
        # splitting into two.
        operation_model = enforcer.model
        enforcer.assert_supported("chat")
        if request.tools:
            enforcer.assert_supported("tools")
        if stream:
            enforcer.assert_supported("streaming")
        if self._messages_contain_images(request.messages):
            enforcer.assert_supported("vision")

        if stream:
            return cast(
                Iterator[StreamChunk],
                self._run_stream_pipeline(
                    "chat",
                    operation_provider,
                    operation_model,
                    request,
                    lambda ctx: self._execute_streaming_chat(
                        cast(UnifiedRequest, ctx.request), stream_callback
                    ),
                ),
            )
        return cast(
            UnifiedResponse,
            self._run_pipeline(
                "chat",
                operation_provider,
                operation_model,
                request,
                lambda ctx: self._execute_chat(cast(UnifiedRequest, ctx.request)),
            ),
        )

    def embed(
        self,
        text: str | list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> EmbeddingsResponse:
        """
        Generate embeddings for one or more input texts.

        Routed through the registered middleware pipeline (if any).

        Args:
            text: A single text or a list of texts to embed.
            provider: Target provider (uses default if not specified).
            model: Embedding model name (uses default if not specified).

        Returns:
            An ``EmbeddingsResponse`` with one vector per input text.
        """
        provider_lower = (provider or self._default_provider).lower()
        enforcer = self._enforcer(provider_lower, model, "embeddings")
        enforcer.assert_supported("embeddings")
        return cast(
            EmbeddingsResponse,
            self._run_pipeline(
                "embed",
                provider_lower,
                enforcer.model,
                None,
                lambda _ctx: self._embed_request(text, provider, model),
            ),
        )

    def _embed_request(
        self,
        text: str | list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> EmbeddingsResponse:
        """
        Generate embeddings for one or more input texts.

        Args:
            text: A single text or a list of texts to embed.
            provider: Target provider (uses default if not specified).
            model: Embedding model name (uses default if not specified).

        Returns:
            An ``EmbeddingsResponse`` with one vector per input text.
        """
        provider_lower = (provider or self._default_provider).lower()
        config, resolved_id, model_info = self._resolve(provider_lower, model, "embeddings")

        if not model_info.capabilities.embeddings:
            raise FeatureNotSupportedError(
                feature="embeddings",
                provider=provider_lower,
                model=resolved_id,
            )

        api_key = self._get_api_key(config)
        if not api_key:
            raise ValueError(
                f"API key not found for provider {provider_lower}. "
                f"Set {config.api_key_env_var} environment variable or pass api_key."
            )

        adapter = self._get_adapter(provider_lower)
        adapter.authenticate({"api_key": api_key})

        texts = [text] if isinstance(text, str) else list(text)
        body = adapter.format_embed_request(resolved_id, texts)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                f"{config.base_url}{adapter.embed_path}",
                headers=headers,
                json=body,
                timeout=self._timeout_for(config),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, provider_lower) from e
        except httpx.TimeoutException as e:
            raise UAITimeoutError(f"Request timeout: {e}", provider=provider_lower) from e
        except httpx.RequestError as e:
            raise UAINetworkError(f"Network error: {e}", provider=provider_lower) from e

        result = adapter.parse_embed_response(response.json(), resolved_id)
        if result.provider is None:
            result.provider = provider_lower
        return result

    def rerank(
        self,
        query: str,
        documents: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> RerankResponse:
        """
        Rank documents by relevance to a query.

        Routed through the registered middleware pipeline (if any).

        Args:
            query: The query text to rank documents against.
            documents: The candidate documents to rerank.
            provider: Provider name (uses default if not specified).
            model: Rerank model name (uses default if not specified).

        Returns:
            A RerankResponse with documents ordered by descending relevance.
        """
        provider_lower = (provider or self._default_provider).lower()
        enforcer = self._enforcer(provider_lower, model, "rerank")
        enforcer.assert_supported("rerank")
        return cast(
            RerankResponse,
            self._run_pipeline(
                "rerank",
                provider_lower,
                enforcer.model,
                None,
                lambda _ctx: self._rerank_request(query, documents, provider, model),
            ),
        )

    def _rerank_request(
        self,
        query: str,
        documents: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> RerankResponse:
        """
        Rank documents by relevance to a query.

        Args:
            query: The query text to rank documents against.
            documents: The candidate documents to rerank.
            provider: Provider name (uses default if not specified).
            model: Rerank model name (uses default if not specified).

        Returns:
            A RerankResponse with documents ordered by descending relevance.
        """
        provider_lower = (provider or self._default_provider).lower()
        config, resolved_id, model_info = self._resolve(provider_lower, model, "rerank")

        if not model_info.capabilities.rerank:
            raise FeatureNotSupportedError(
                feature="rerank",
                provider=provider_lower,
                model=resolved_id,
            )

        api_key = self._get_api_key(config)
        if not api_key:
            raise ValueError(
                f"API key not found for provider {provider_lower}. "
                f"Set {config.api_key_env_var} environment variable or pass api_key."
            )

        adapter = self._get_adapter(provider_lower)
        adapter.authenticate({"api_key": api_key})

        body = self._format_rerank_body(adapter, resolved_id, query, documents)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                f"{config.base_url}{adapter.rerank_path}",
                headers=headers,
                json=body,
                timeout=self._timeout_for(config),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, provider_lower) from e
        except httpx.TimeoutException as e:
            raise UAITimeoutError(f"Request timeout: {e}", provider=provider_lower) from e
        except httpx.RequestError as e:
            raise UAINetworkError(f"Network error: {e}", provider=provider_lower) from e

        result = adapter.parse_rerank_response(response.json(), resolved_id)
        if result.provider is None:
            result.provider = provider_lower
        return result

    def _format_rerank_body(
        self, adapter: BaseProviderAdapter, model: str, query: str, documents: list[str]
    ) -> dict[str, Any]:
        """Build the rerank request body via the adapter's format method."""
        return adapter.format_rerank_request(model, query, documents)

    def _execute_chat(self, request: UnifiedRequest) -> UnifiedResponse:
        """Execute a non-streaming chat request."""
        provider_lower = (request.provider or self._default_provider).lower()
        config, resolved_id, model_info = self._resolve(provider_lower, request.model, "chat")

        # Post-middleware safety net: the boundary enforcer already gated
        # the original request, but before_request hooks may have swapped
        # the provider/model, so re-verify against the resolved model.
        if not model_info.capabilities.chat:
            raise FeatureNotSupportedError(
                feature="chat",
                provider=provider_lower,
                model=resolved_id,
            )

        # Get API key
        api_key = self._get_api_key(config)
        if not api_key:
            raise ValueError(
                f"API key not found for provider {provider_lower}. "
                f"Set {config.api_key_env_var} environment variable or pass api_key."
            )

        # Build headers and body
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        adapter = self._get_adapter(provider_lower)
        body = self._build_request_body(request, resolved_id, adapter)

        # Make API request
        start_time = time.time()

        try:
            response = httpx.post(
                f"{config.base_url}{adapter.chat_path}",
                headers=headers,
                json=body,
                timeout=self._timeout_for(config),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, provider_lower) from e
        except httpx.TimeoutException as e:
            raise UAITimeoutError(f"Request timeout: {e}", provider=provider_lower) from e
        except httpx.RequestError as e:
            raise UAINetworkError(f"Network error: {e}", provider=provider_lower) from e

        data = response.json()
        return self._parse_chat_response(data, provider_lower, resolved_id, start_time, request)

    def _build_request_body(
        self,
        request: UnifiedRequest,
        model_id: str,
        adapter: BaseProviderAdapter,
    ) -> dict[str, Any]:
        """
        Build the provider wire body by delegating to *adapter*.

        The client used to hand-roll an OpenAI-shaped body here, which meant
        every adapter's ``format_request`` was dead code: DeepSeek's thinking
        parameter, MiniMax's content-block flattening and the
        ``frequency_penalty``/``presence_penalty``/``user`` fields all existed
        in the adapters but never reached the wire. Routing through the
        adapter is what makes those provider-specific translations real.

        The request handed to the adapter is a copy carrying the *canonical*
        model id, so aliases are dereferenced before any adapter compares
        ``request.model`` against a known id.
        """
        messages = list(request.messages)
        if request.output_schema is not None:
            # Module 1.3.2 — nudge the model toward schema-conforming JSON by
            # injecting the JSON Schema as a system instruction.
            messages = [
                ChatMessage(
                    role=Role.SYSTEM,
                    content=build_schema_prompt(request.output_schema),
                ),
                *messages,
            ]

        outbound = request.model_copy(update={"model": model_id, "messages": messages})
        return adapter.format_request(outbound)

    def _execute_streaming_chat(
        self,
        request: UnifiedRequest,
        callback: Callable[[StreamChunk], None] | None = None,
    ) -> Iterator[StreamChunk]:
        """Execute a streaming chat request."""
        provider_lower = (request.provider or self._default_provider).lower()
        config, resolved_id, model_info = self._resolve(provider_lower, request.model, "chat")

        # Post-middleware safety net (see ``_execute_chat``).
        if not model_info.capabilities.streaming:
            raise FeatureNotSupportedError(
                feature="streaming",
                provider=provider_lower,
                model=resolved_id,
            )

        # Get API key
        api_key = self._get_api_key(config)
        if not api_key:
            raise ValueError(f"API key not found for provider {provider_lower}")

        # Build headers and body
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        adapter = self._get_adapter(provider_lower)
        body = self._build_request_body(request, resolved_id, adapter)
        body["stream"] = True

        start_time = time.time()
        first_chunk = True
        finish_reason = None
        seen_ids = set()
        # Module 1.3.2 — accumulate content deltas so the assembled payload
        # can be validated against ``output_schema`` when the stream ends.
        output_schema = request.output_schema
        content_parts: list[str] = []

        def _finalize_parsed() -> Any:
            """Validate accumulated content against output_schema, if set."""
            if output_schema is None or not content_parts:
                return None
            return parse_structured_output(
                "".join(content_parts), output_schema, provider=provider_lower
            )

        finalized = False

        try:
            with httpx.stream(
                "POST",
                f"{config.base_url}{adapter.chat_path}",
                headers=headers,
                json=body,
                timeout=self._timeout_for(config),
            ) as response:
                if response.status_code != 200:
                    raise self._handle_http_error(
                        httpx.HTTPStatusError("Error", request=response.request, response=response),
                        provider_lower,
                    )

                for line in response.iter_lines():
                    if not line:
                        continue

                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line

                    if line_str.startswith("data: "):
                        line_str = line_str[6:]

                    if line_str.strip() in ("[DONE]", "data: [DONE]"):
                        yield StreamChunk(is_final=True, parsed=_finalize_parsed())
                        finalized = True
                        break

                    if not line_str.strip():
                        continue

                    try:
                        chunk_data = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    if not chunk_data.get("choices"):
                        continue

                    choice = chunk_data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")

                    # Record TTFT on first content chunk
                    if first_chunk and content:
                        ttft_ms = (time.time() - start_time) * 1000
                        first_chunk = False
                    else:
                        ttft_ms = None

                    # Extract finish reason
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    # Extract tool calls
                    tool_calls = None
                    if delta.get("tool_calls"):
                        from uai.models import FunctionCall, ToolCall

                        tool_calls = [
                            ToolCall(
                                id=tc.get("id", ""),
                                type="function",
                                function=FunctionCall(
                                    name=tc.get("function", {}).get("name", ""),
                                    arguments=tc.get("function", {}).get("arguments", "{}"),
                                ),
                            )
                            for tc in delta["tool_calls"]
                            if tc.get("id")
                        ]

                    # Get ID
                    chunk_id = chunk_data.get("id")
                    actual_id = chunk_id if chunk_id and chunk_id not in seen_ids else None
                    if chunk_id:
                        seen_ids.add(chunk_id)

                    # Extract usage (some providers send it on the final chunk)
                    usage = None
                    usage_dict = chunk_data.get("usage")
                    if usage_dict:
                        usage = UsageMetrics(
                            input_tokens=usage_dict.get("prompt_tokens", 0),
                            output_tokens=usage_dict.get("completion_tokens", 0),
                            cache_read_tokens=usage_dict.get("cache_read_input_tokens"),
                            cache_write_tokens=usage_dict.get("cache_creation_input_tokens"),
                        )

                    if content and output_schema is not None:
                        content_parts.append(content)

                    chunk = StreamChunk(
                        content=content if content else None,
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                        id=actual_id,
                        model=resolved_id,
                        provider=provider_lower,
                        usage=usage,
                        is_final=finish_reason is not None,
                        ttft_ms=ttft_ms,
                    )

                    if finish_reason and output_schema is not None:
                        # Validate before emitting the terminal chunk so a
                        # schema failure surfaces as ResponseParsingError.
                        chunk.parsed = _finalize_parsed()
                        finalized = True

                    if callback:
                        callback(chunk)

                    yield chunk

                    if finish_reason:
                        break

                # A provider may end the stream without [DONE] or an explicit
                # finish_reason — validate accumulated content regardless.
                if output_schema is not None and content_parts and not finalized:
                    yield StreamChunk(is_final=True, parsed=_finalize_parsed())
        except httpx.RequestError as e:
            raise UAINetworkError(
                f"Network error during streaming: {e}", provider=provider_lower
            ) from e

    def _parse_chat_response(
        self,
        data: dict[str, Any],
        provider: str,
        model: str,
        start_time: float,
        request: UnifiedRequest | None = None,
    ) -> UnifiedResponse:
        """Parse a chat completion response into UnifiedResponse."""
        choices = data.get("choices", [])
        if not choices:
            raise UAIError("No choices in response")

        choice = choices[0]
        message = choice.get("message", {})

        content = message.get("content", "")

        # Parse tool calls
        tool_calls = None
        if message.get("tool_calls"):
            from uai.models import FunctionCall, ToolCall

            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type="function",
                    function=FunctionCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    ),
                )
                for tc in message["tool_calls"]
                if tc.get("type") == "function"
            ]

        # Map finish reason
        finish_reason_raw = choice.get("finish_reason", "stop")
        finish_reason_map = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "function_call": FinishReason.FUNCTION_CALL,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        finish_reason = finish_reason_map.get(finish_reason_raw, FinishReason.STOP)

        # Get usage
        usage_data = data.get("usage", {})
        if not usage_data and choices and "usage" in choices[0]:
            usage_data = choices[0]["usage"]

        usage = UsageMetrics(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens"),
            cache_write_tokens=usage_data.get("cache_creation_input_tokens"),
        )

        # Module 1.3.2 — structured output validation on the way back.
        parsed = None
        if request is not None and request.output_schema is not None and content:
            parsed = parse_structured_output(content, request.output_schema, provider=provider)

        return UnifiedResponse(
            id=data.get("id"),
            provider=provider,
            model=model,
            content=content if content else None,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
            parsed=parsed,
            raw=data,
        )

    @staticmethod
    def _parse_retry_after(response: httpx.Response | None) -> float | None:
        """Parse the ``Retry-After`` header (seconds) if present."""
        if response is None:
            return None
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _handle_http_error(self, error: httpx.HTTPStatusError, provider: str) -> UAIError:
        """Translate HTTP errors into appropriate SDK exceptions."""
        status_code = error.response.status_code if error.response else 0
        response_body = error.response.text if error.response else None
        retry_after = self._parse_retry_after(error.response)

        if status_code == 401:
            raise UAIAuthenticationError(
                f"Authentication failed for provider '{provider}': {response_body}",
                provider=provider,
                status_code=status_code,
                response_body=response_body,
            ) from error
        elif status_code == 429:
            raise UAIRateLimitError(
                f"Rate limit exceeded for provider '{provider}': {response_body}",
                provider=provider,
                status_code=status_code,
                response_body=response_body,
                retry_after=retry_after,
            ) from error
        elif status_code >= 500:
            raise UAIError(
                f"Server error from provider '{provider}': {response_body}",
                provider=provider,
                status_code=status_code,
                response_body=response_body,
            ) from error
        else:
            raise UAIError(
                f"HTTP error {status_code} from provider '{provider}': {response_body}",
                provider=provider,
                status_code=status_code,
                response_body=response_body,
            ) from error
