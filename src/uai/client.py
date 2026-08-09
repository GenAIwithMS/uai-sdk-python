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
from typing import Any, Callable, cast

import httpx

from uai.adapters.base_adapter import BaseProviderAdapter
from uai.enforcer import CapabilityMatrixEnforcer
from uai.exceptions import (
    FeatureNotSupportedError,
    UAIAuthenticationError,
    UAIError,
    UAINetworkError,
    UAIRateLimitError,
    UAITimeoutError,
)
from uai.middleware.base import BaseMiddleware, MiddlewareContext
from uai.middleware.engine import MiddlewareEngine
from uai.models import (
    ChatMessage,
    EmbeddingsResponse,
    FinishReason,
    ImageContent,
    RerankResponse,
    StreamChunk,
    ToolDefinition,
    UnifiedRequest,
    UnifiedResponse,
    UsageMetrics,
)
from uai.registry import apply_env_overrides, get_provider_config
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

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "deepseek",
        model: str | None = None,
        credentials: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        """
        Initialize the UniversalAI client.

        Args:
            api_key: Default API key (can be overridden by provider-specific env vars).
            provider: Default provider to use (e.g., 'deepseek', 'qwen').
            model: Default model to use.
            credentials: Provider-specific credential dictionary.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self._all_configs = apply_env_overrides()

        provider_lower = provider.lower()
        if provider_lower not in self._all_configs:
            available = ", ".join(self._all_configs.keys())
            raise ValueError(
                f"Provider '{provider}' is not available. Available providers: {available}"
            )

        self._default_provider = provider_lower
        base_config = get_provider_config(self._default_provider)

        self._config: ProviderConfig = base_config.model_copy()

        if timeout is not None:
            self._config.timeout = timeout
        if max_retries is not None:
            self._config.max_retries = max_retries

        if credentials is not None:
            self._credentials = credentials
        elif api_key is not None:
            self._credentials = {"api_key": api_key}
        else:
            api_key_env = self._config.api_key_env_var
            env_key = os.environ.get(api_key_env) if api_key_env else None
            self._credentials = {"api_key": env_key} if env_key else {}

        self._default_model = model or self._config.default_model

        self._adapters: dict[str, BaseProviderAdapter] = {}
        self._engine = MiddlewareEngine()

        logger.debug(
            f"UniversalAI client initialized with provider={self._default_provider}, "
            f"model={self._default_model}"
        )

    def use(self, middleware: BaseMiddleware | list[BaseMiddleware]) -> UniversalAI:
        """
        Register one or more middleware instances (opt-in pipeline).

        Middleware hooks run in registration order for ``before_request``
        and in reverse order for ``after_response``/``on_error``.

        Args:
            middleware: A single middleware or a list of middleware.

        Returns:
            The client, for chaining.
        """
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
        return self._engine.run(operation, provider, model, request, execute_fn)

    def _run_stream_pipeline(
        self,
        operation: str,
        provider: str,
        model: str,
        request: Any,
        stream_fn: Callable[[MiddlewareContext], Any],
    ) -> Any:
        """Run before -> execute around a streaming callable; after on finish (Module 1.4.1)."""
        return self._engine.run_stream(operation, provider, model, request, stream_fn)

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

    def _get_api_key(self, config: ProviderConfig) -> str | None:
        """Get API key from credentials or environment."""
        api_key = self._credentials.get("api_key")
        if api_key is not None:
            return str(api_key)

        api_key_env = config.api_key_env_var
        result = os.environ.get(api_key_env) if api_key_env else None
        return str(result) if result is not None else None

    def _enforcer(self, provider_lower: str, model_id: str) -> CapabilityMatrixEnforcer:
        """
        Build a capability enforcer for the active provider/model/adapter.

        The enforcer merges the registry model capabilities with the
        adapter's ``capabilities()`` matrix (Module 1.3.1).
        """
        config = self._resolve_model(provider_lower, model_id)
        adapter = self._get_adapter(provider_lower)
        return CapabilityMatrixEnforcer(
            provider_lower,
            model_id,
            adapter=adapter,
            config=config,
        )

    def supports(self, feature: str, provider: str | None = None, model: str | None = None) -> bool:
        """
        Pre-flight check: does the resolved provider/model support *feature*?

        Args:
            feature: Capability name (e.g. ``'vision'``, ``'tools'``).
            provider: Provider name (defaults to the client's provider).
            model: Model id or alias (defaults to the client's model).

        Returns:
            ``True`` if the feature is supported by both the registry model
            capabilities and the active adapter's matrix.
        """
        provider_lower = (provider or self._default_provider).lower()
        model_id = model or self._default_model
        return self._enforcer(provider_lower, model_id).supports(feature)

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

        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(request, key) and value is not None:
                setattr(request, key, value)

        operation_provider = request.provider or self._default_provider
        operation_model = request.model or self._default_model

        # Module 1.3.1 — capability matrix enforcement.  Halt before any
        # middleware or network work if the requested features are unsupported.
        enforcer = self._enforcer(operation_provider.lower(), operation_model)
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
        self._enforcer(provider_lower, model or self._default_model).assert_supported("embeddings")
        return cast(
            EmbeddingsResponse,
            self._run_pipeline(
                "embed",
                provider_lower,
                model or self._default_model,
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
        config = self._resolve_model(provider_lower, model or self._default_model)

        model_id = model or self._default_model
        resolved_id, model_info = self._resolve_model_info(config, model_id)

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
                timeout=config.timeout,
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
        self._enforcer(provider_lower, model or self._default_model).assert_supported("rerank")
        return cast(
            RerankResponse,
            self._run_pipeline(
                "rerank",
                provider_lower,
                model or self._default_model,
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
        config = self._resolve_model(provider_lower, model or self._default_model)

        model_id = model or self._default_model
        resolved_id, model_info = self._resolve_model_info(config, model_id)

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
                timeout=config.timeout,
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

    def _resolve_model(self, provider_lower: str, model_id: str) -> ProviderConfig:
        """Get provider config and resolve model."""
        config = self._all_configs.get(provider_lower)
        if config:
            return config
        return get_provider_config(provider_lower)

    def _resolve_model_info(
        self, config: ProviderConfig, model_id: str
    ) -> tuple[str, ProviderModel]:
        """Resolve model ID and return (resolved_id, model_info)."""
        model_info = config.models.get(model_id)
        resolved_id = model_id

        if not model_info:
            # Try alias lookup
            for mid, m in config.models.items():
                if model_id in m.aliases:
                    model_info = m
                    resolved_id = mid
                    break

        if not model_info:
            raise ValueError(f"Model '{model_id}' not found")

        return resolved_id, model_info

    def _execute_chat(self, request: UnifiedRequest) -> UnifiedResponse:
        """Execute a non-streaming chat request."""
        provider_lower = (request.provider or self._default_provider).lower()
        config = self._resolve_model(provider_lower, request.model or self._default_model)

        model_id = request.model or self._default_model
        resolved_id, model_info = self._resolve_model_info(config, model_id)

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

        body = self._build_request_body(request, resolved_id)

        # Make API request
        start_time = time.time()

        try:
            response = httpx.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=config.timeout,
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

    def _build_request_body(self, request: UnifiedRequest, model_id: str) -> dict[str, Any]:
        """Build the request body for the provider API."""
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        if request.output_schema is not None:
            # Module 1.3.2 — nudge the model toward schema-conforming JSON by
            # injecting the JSON Schema as a system instruction.
            messages = [
                {"role": "system", "content": build_schema_prompt(request.output_schema)},
                *messages,
            ]
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
        }

        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]
        if request.tools:
            body["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
            if request.tool_choice:
                body["tool_choice"] = request.tool_choice.value

        return body

    def _execute_streaming_chat(
        self,
        request: UnifiedRequest,
        callback: Callable[[StreamChunk], None] | None = None,
    ) -> Iterator[StreamChunk]:
        """Execute a streaming chat request."""
        provider_lower = (request.provider or self._default_provider).lower()
        config = self._resolve_model(provider_lower, request.model or self._default_model)

        model_id = request.model or self._default_model
        resolved_id, model_info = self._resolve_model_info(config, model_id)

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

        body = self._build_request_body(request, resolved_id)
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
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=config.timeout,
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
