"""
Abstract base adapter contract for the Universal AI Provider SDK.

This module defines the interface that all provider adapters must implement.
Each adapter translates between the SDK's unified request/response models
and the provider's specific API format.

The architecture mandates that provider-specific business logic, request
formatting quirks, and error translations reside entirely within explicit
source code, completely rejecting the use of YAML-based configuration
files for behavioral logic.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Iterator
from typing import Any

from uai.exceptions import FeatureNotSupportedError
from uai.models import (
    EmbeddingsResponse,
    RerankResponse,
    StreamChunk,
    UnifiedRequest,
    UnifiedResponse,
)


class BaseProviderAdapter(abc.ABC):
    """
    Abstract base class for provider adapters.

    Every adapter must provide concrete implementations for all lifecycle
    methods defined here. The base class enforces strict separation of
    concerns: request formatting, response parsing, and error translation
    are all provider-specific.

    Attributes:
        provider_name: The canonical provider name (e.g., 'deepseek', 'qwen').
    """

    provider_name: str = ""

    # Endpoint paths used for multimodal / non-chat features.  Providers
    # override these when their wire paths differ from the defaults.
    embed_path: str = "/embeddings"
    rerank_path: str = "/rerank"

    @abc.abstractmethod
    def authenticate(self, credentials: dict[str, Any]) -> None:
        """
        Set up authentication for the provider.

        Called once during adapter initialization. Implementations should
        configure HTTP client headers, tokens, or credentials as needed.

        Args:
            credentials: Dictionary containing authentication information.
                        May include 'api_key', 'bearer_token', or provider-specific keys.
        """
        ...

    @abc.abstractmethod
    def format_request(self, request: UnifiedRequest) -> dict[str, Any]:
        """
        Translate a UnifiedRequest into the provider's wire format.

        Args:
            request: The normalized, provider-agnostic request.

        Returns:
            A dictionary ready to be JSON-serialized and sent to the provider.
        """
        ...

    @abc.abstractmethod
    def parse_response(self, response: dict[str, Any], request: UnifiedRequest) -> UnifiedResponse:
        """
        Translate a provider's response into a UnifiedResponse.

        Args:
            response: The raw JSON response from the provider.
            request: The original request (for context and metadata).

        Returns:
            A normalized UnifiedResponse object.
        """
        ...

    def handle_streaming(
        self,
        response: Any,
        request: UnifiedRequest,
    ) -> Iterator[StreamChunk]:
        """
        Parse streaming responses from the provider.

        Default implementation handles Server-Sent Events (SSE) format.
        Subclasses may override for provider-specific streaming formats.

        Args:
            response: The HTTP response with streaming content.
            request: The original request (for metadata).

        Yields:
            StreamChunk objects for each chunk received.
        """
        ttft_recorded = False

        try:
            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8") if isinstance(line, bytes) else line

                if line_str.startswith("data: "):
                    line_str = line_str[6:]

                if line_str.strip() in ("data: [DONE]", "[DONE]"):
                    yield StreamChunk(is_final=True)
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

                # Record TTFT on first content chunk
                if not ttft_recorded:
                    content_sample = delta.get("content", "")
                    if content_sample:
                        yield StreamChunk(ttft_ms=0.0, is_final=False)
                        ttft_recorded = True

                # Extract content delta
                content = delta.get("content", "")

                # Extract finish reason
                finish_reason = choice.get("finish_reason")

                # Extract tool calls
                tool_calls = None
                tool_calls_data = delta.get("tool_calls")
                if tool_calls_data:
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
                        for tc in tool_calls_data
                    ]

                # Extract usage
                usage = None
                usage_dict = chunk_data.get("choices", [{}])[0].get("usage")
                if usage_dict:
                    from uai.models import UsageMetrics

                    usage = UsageMetrics(
                        input_tokens=usage_dict.get("prompt_tokens", 0),
                        output_tokens=usage_dict.get("completion_tokens", 0),
                        cache_read_tokens=usage_dict.get("cache_read_input_tokens"),
                        cache_write_tokens=usage_dict.get("cache_creation_input_tokens"),
                    )

                chunk = StreamChunk(
                    content=content if content else None,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    usage=usage,
                    is_final=finish_reason is not None,
                )

                yield chunk

                if finish_reason:
                    break

        except Exception as e:
            from uai.exceptions import UAINetworkError

            raise UAINetworkError(f"Streaming failed: {e}") from e

    @abc.abstractmethod
    def translate_error(self, status_code: int, error_body: Any) -> Exception:
        """
        Translate provider-specific errors into SDK exceptions.

        Args:
            status_code: The HTTP status code.
            error_body: The error response body.

        Returns:
            An appropriate UAIError subclass exception.
        """
        ...

    @abc.abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """
        Return the capability matrix for this adapter.

        Returns a dictionary mapping capability names to boolean support
        status. The SDK will use this to gate feature requests and raise
        FeatureNotSupportedError when appropriate.

        Returns:
            Dict mapping capability names to bool (e.g., {'chat': True, 'vision': False}).
        """
        ...

    # -- Embeddings --------------------------------------------------------

    def format_embed_request(self, model: str, texts: list[str]) -> dict[str, Any]:
        """
        Translate an embedding request into the provider's wire format.

        The default implementation targets the OpenAI-compatible
        ``POST /embeddings`` schema, which the majority of providers share.
        Adapters whose embedding endpoints differ may override this method.

        Args:
            model: The embedding model name.
            texts: The input texts to embed.

        Returns:
            A dictionary ready to be JSON-serialized and sent to the provider.
        """
        return {"model": model, "input": texts}

    def parse_embed_response(
        self, response: dict[str, Any], model: str | None
    ) -> EmbeddingsResponse:
        """
        Translate a provider's embedding response into EmbeddingsResponse.

        The default implementation covers the OpenAI-compatible
        ``{"data": [{"embedding": [...], "index": n}], "usage": {...}}``
        schema shared by most providers.

        Args:
            response: The raw JSON response from the provider.
            model: The model name used for the request.

        Returns:
            A normalized EmbeddingsResponse object.
        """
        from uai.models import EmbeddingResult, UsageMetrics

        usage_raw = response.get("usage") or {}
        data = response.get("data") or []
        vectors: list[EmbeddingResult] = []
        for item in data:
            vectors.append(
                EmbeddingResult(
                    values=item.get("embedding", []),
                    dimension=len(item.get("embedding", [])),
                    index=item.get("index", 0),
                )
            )
        usage = UsageMetrics(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
        )
        return EmbeddingsResponse(
            vectors=vectors,
            model=model or response.get("model"),
            provider=self.provider_name,
            usage=usage,
            raw=response,
        )

    # -- Rerank ------------------------------------------------------------

    def format_rerank_request(self, model: str, query: str, documents: list[str]) -> dict[str, Any]:
        """
        Translate a rerank request into the provider's wire format.

        Providers vary widely in rerank schemas, so the base implementation
        raises FeatureNotSupportedError. Adapters for rerank-capable providers
        must override this method.

        Args:
            model: The rerank model name.
            query: The query text against which documents are scored.
            documents: The candidate documents to rerank.

        Returns:
            A dictionary ready to be JSON-serialized and sent to the provider.
        """
        raise FeatureNotSupportedError(feature="rerank", provider=self.provider_name)

    def parse_rerank_response(
        self, response: dict[str, Any], model: str | None = None
    ) -> RerankResponse:
        """
        Parse a provider's rerank response into RerankResponse.

        Providers vary widely in rerank response schemas, so the default
        raises FeatureNotSupportedError. Adapters for rerank-capable
        providers must override this method.
        """
        raise FeatureNotSupportedError(feature="rerank", provider=self.provider_name)
