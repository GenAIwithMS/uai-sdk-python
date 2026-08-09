"""
Unified data models for the Universal AI Provider SDK.

This module defines the canonical data structures that insulate the
application layer from provider-specific JSON schemas.  Every request
flows through ``UnifiedRequest`` and every response is normalized into
``UnifiedResponse``, regardless of the underlying provider (DeepSeek,
Qwen, GLM, etc.).

The models use Pydantic v2 for strict runtime type enforcement and
validation, failing fast on malformed input rather than silently
producing ambiguous behaviour.

Supported developer input shapes (normalized by the models):

* **Messages** — accept either ``ChatMessage`` instances or raw dicts
  in the OpenAI/Anthropic-style format, e.g.::

      {"role": "user", "content": "Hello"}
      {"role": "assistant", "tool_calls": [...]}
      {"role": "tool", "tool_call_id": "...", "content": "..."}

* **Tools** — accept either ``ToolDefinition`` instances or dicts::

      {"type": "function", "function": {"name": "...", "parameters": {...}}}

* **Structured output** — ``output_schema`` accepts a ``type[BaseModel]``
  (Pydantic) class, which the SDK converts to a JSON Schema for the
  provider and validates the response against on the way back.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

__all__ = [
    "ChatMessage",
    "ContentBlock",
    "EmbeddingResult",
    "EmbeddingsResponse",
    "FinishReason",
    "FunctionCall",
    "FunctionDefinition",
    "ImageContent",
    "ImageURL",
    "RerankResponse",
    "RerankResult",
    "Role",
    "StreamChunk",
    "TextContent",
    "ToolCall",
    "ToolCallMode",
    "ToolDefinition",
    "UnifiedRequest",
    "UnifiedResponse",
    "UsageMetrics",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Roles that a message can play in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"
    FUNCTION = "function"


class FinishReason(str, Enum):
    """
    Standardized finish reasons.

    Regardless of whether the provider returns ``"stop"``, ``"tool_calls"``,
    ``1``, or ``"function_call"``, the SDK normalizes to one of these
    values so application code can branch uniformly.
    """

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    NULL = "null"
    OTHER = "other"


class ToolCallMode(str, Enum):
    """How the provider should use tools, if any are provided."""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


# ---------------------------------------------------------------------------
# Usage metrics
# ---------------------------------------------------------------------------


class UsageMetrics(BaseModel):
    """Token usage metrics for a single request/response pair."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0, description="Number of input (prompt) tokens.")
    output_tokens: int = Field(default=0, ge=0, description="Number of output (completion) tokens.")
    total_tokens: int = Field(
        default=0, ge=0, description="Total tokens (input + output). Auto-computed if 0."
    )
    cache_read_tokens: int | None = Field(
        default=None, ge=0, description="Tokens read from the provider's cache (if supported)."
    )
    cache_write_tokens: int | None = Field(
        default=None, ge=0, description="Tokens written to the provider's cache (if supported)."
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Tokens used for internal reasoning (e.g. DeepSeek reasoner).",
    )

    @model_validator(mode="after")
    def _compute_total(self) -> UsageMetrics:
        """Auto-compute ``total_tokens`` when it is left at 0."""
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens
        return self

    def add(self, other: UsageMetrics) -> UsageMetrics:
        """Merge another ``UsageMetrics`` into this one (in-place additive)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        if other.cache_read_tokens is not None:
            self.cache_read_tokens = (self.cache_read_tokens or 0) + other.cache_read_tokens
        if other.cache_write_tokens is not None:
            self.cache_write_tokens = (self.cache_write_tokens or 0) + other.cache_write_tokens
        if other.reasoning_tokens is not None:
            self.reasoning_tokens = (self.reasoning_tokens or 0) + other.reasoning_tokens
        return self


# ---------------------------------------------------------------------------
# Message content blocks
# ---------------------------------------------------------------------------


class ImageURL(BaseModel):
    """Image URL reference inside an ``ImageContent`` block."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, description="Data URI (base64) or HTTP(S) URL.")
    detail: Literal["auto", "low", "high"] = Field(
        default="auto", description="Resolution hint for vision models."
    )


class TextContent(BaseModel):
    """A text content block within a message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ImageContent(BaseModel):
    """An image content block within a message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


# Discriminated union — Pydantic selects the subclass via the ``type`` field.
ContentBlock = Annotated[
    Union[TextContent, ImageContent],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Tool calls (in assistant messages)
# ---------------------------------------------------------------------------


class FunctionCall(BaseModel):
    """The ``function`` part of a tool call as returned by the provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: str = Field(
        default="",
        description="JSON-encoded argument string returned by the provider. "
        "May be partial in streaming responses.",
    )


class ToolCall(BaseModel):
    """A single tool/function call made by the model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Unique identifier for the tool call.")
    type: Literal["function"] = "function"
    function: FunctionCall = Field(description="The function being called.")

    def get_arguments(self) -> dict[str, Any]:
        """
        Parse the raw ``arguments`` JSON string into a Python dict.

        :returns: Parsed arguments, or an empty dict if the string is empty.
        :raises json.JSONDecodeError: If the arguments string is malformed.
        """
        stripped = self.function.arguments.strip()
        if not stripped:
            return {}
        return json.loads(stripped)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Tool definitions (in the request)
# ---------------------------------------------------------------------------


class FunctionDefinition(BaseModel):
    """Definition of a function that the model can call."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = Field(
        default=None, description="Human-readable description of the function."
    )
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema describing the function's parameters.",
    )
    strict: bool | None = Field(
        default=None,
        description="Whether to enable strict schema adherence. "
        "``None`` lets the provider use its default.",
    )


class ToolDefinition(BaseModel):
    """A tool callable by the model, in the OpenAI-compatible format."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: FunctionDefinition


# ---------------------------------------------------------------------------
# Chat message
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """
    A single message in a conversation.

    ``content`` may be a plain string (most common) or a list of
    :class:`ContentBlock` objects (for multimodal messages).  When
    ``role`` is ``"assistant"`` the message may carry ``tool_calls``;
    when ``role`` is ``"tool"`` it must carry a ``tool_call_id``.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role = Field(description="The role of the message author.")
    content: str | list[ContentBlock] | None = Field(
        default=None,
        description="The message content. A string for simple text, a list of "
        "content blocks for multimodal messages, or None when the message "
        "consists solely of tool_calls.",
    )
    name: str | None = Field(
        default=None,
        description="Optional speaker name (e.g. for 'user' messages from a specific participant).",
    )
    # Assistant-side fields
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Tool calls requested by the assistant (role='assistant').",
    )
    # Tool-side fields
    tool_call_id: str | None = Field(
        default=None,
        description="ID of the tool call this message responds to (role='tool').",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_before(cls, data: Any) -> Any:
        """Accept plain dicts and normalize the ``role`` field."""
        if isinstance(data, dict) and data.get("content") is None and data.get("role") == "tool":
            data["content"] = ""
        return data


# ---------------------------------------------------------------------------
# Unified request
# ---------------------------------------------------------------------------


class UnifiedRequest(BaseModel):
    """
    The canonical, provider-agnostic request model.

    The client normalizes arbitrary developer inputs (prompt strings,
    message dicts, tool dicts) into this strict schema *before* passing
    it through the middleware pipeline and to the provider adapter.
    Adapters then translate the unified model into each provider's own
    wire format.

    By funneling all provider-specific data through this model, the core
    application logic remains insulated from vendor-side API mutations.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # --- Routing ----------------------------------------------------------
    provider: str | None = Field(
        default=None,
        description="Target provider name (e.g. 'deepseek'). If None, falls back to the client's default.",
    )
    model: str | None = Field(
        default=None,
        description="Provider-specific model ID (e.g. 'deepseek-chat'). "
        "If None, the provider's default model is used.",
    )

    # --- Core payload -----------------------------------------------------
    messages: list[ChatMessage] = Field(
        description="Conversation messages in chronological order.",
    )

    # --- Generation parameters --------------------------------------------
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum number of output tokens to generate."
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature."
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter."
    )
    stop: list[str] | str | None = Field(
        default=None,
        description="Up to 4 stop sequences where generation should halt.",
    )
    frequency_penalty: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Penalize new tokens based on frequency."
    )
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Penalize new tokens based on presence."
    )

    # --- Tools & structured output -------------------------------------
    tools: list[ToolDefinition] | None = Field(
        default=None, description="Functions/tools the model may invoke."
    )
    tool_choice: ToolCallMode | None = Field(
        default=None,
        description="Control how the model uses tools: 'none', 'auto', or 'required'.",
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description="Pydantic model class used to enforce structured JSON output. "
        "The SDK converts this to a JSON Schema and validates the response against it.",
    )

    # --- Request behaviour ------------------------------------------------
    stream: bool = Field(default=False, description="Whether to stream the response.")

    # --- Metadata ---------------------------------------------------------
    user: str | None = Field(
        default=None,
        description="End-user identifier for abuse monitoring.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for middleware and telemetry.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        """
        Accept loose developer inputs and coerce them into the strict model.

        * ``messages`` — list of dicts are converted to ``ChatMessage``.
        * ``tools`` — list of dicts are converted to ``ToolDefinition``.
        * ``stop`` — a single string is wrapped in a list.
        """
        if not isinstance(data, dict):
            return data

        # Normalize messages
        messages = data.get("messages")
        if messages is not None:
            normalized: list[Any] = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    normalized.append(msg)
                elif isinstance(msg, dict):
                    normalized.append(ChatMessage(**msg))
                elif isinstance(msg, str):
                    normalized.append(ChatMessage(role=Role.USER, content=msg))
                else:
                    raise TypeError(
                        f"Message must be a ChatMessage, dict, or str, got {type(msg).__name__}"
                    )
            data["messages"] = normalized

        # Normalize tools
        tools = data.get("tools")
        if tools is not None:
            normalized_tools: list[ToolDefinition] = []
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    normalized_tools.append(tool)
                elif isinstance(tool, dict):
                    normalized_tools.append(ToolDefinition(**tool))
                else:
                    raise TypeError(
                        f"Tool must be a ToolDefinition or dict, got {type(tool).__name__}"
                    )
            data["tools"] = normalized_tools

        # Normalize stop: string → list[str]
        stop = data.get("stop")
        if isinstance(stop, str):
            data["stop"] = [stop]

        return data

    @model_validator(mode="after")
    def _validate_after(self) -> UnifiedRequest:
        # tool_choice only makes sense when tools are provided
        if (
            self.tool_choice is not None
            and self.tool_choice != ToolCallMode.NONE
            and self.tools is None
        ):
            raise ValueError("tool_choice can only be 'auto' or 'required' when tools are provided")

        # output_schema requires the model to support structured output;
        # we don't have the provider info here, so we set a flag instead.
        # The client/adapter will check the capability matrix.
        return self


# ---------------------------------------------------------------------------
# Unified response
# ---------------------------------------------------------------------------


class UnifiedResponse(BaseModel):
    """
    The canonical, provider-agnostic response model.

    Regardless of the underlying provider's idiosyncratic JSON response,
    the developer always receives this standardized object containing:

    * ``content`` — the parsed text output (or ``None`` if the model
      emitted only tool calls).
    * ``usage`` — exhaustive token usage metrics.
    * ``finish_reason`` — a standardized finish reason enum.
    * ``tool_calls`` — parsed tool calls, if any.
    * ``parsed`` — the validated structured-output model, if
      ``output_schema`` was supplied.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    id: str | None = Field(
        default=None,
        description="Unique identifier for the completion, from the provider.",
    )
    provider: str | None = Field(
        default=None,
        description="Provider that served the request (e.g. 'deepseek').",
    )
    model: str | None = Field(
        default=None,
        description="Actual model that served the request (may differ from request model).",
    )
    content: str | None = Field(
        default=None,
        description="The generated text output.",
    )
    finish_reason: FinishReason = Field(
        default=FinishReason.STOP,
        description="Standardized finish reason.",
    )
    usage: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token usage metrics for this completion.",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Tool calls made by the model, if any.",
    )
    parsed: BaseModel | None = Field(
        default=None,
        description="Validated structured output (when ``output_schema`` was provided).",
    )
    raw: dict[str, Any] | None = Field(
        default=None,
        description="The original provider response payload, for debugging.",
    )


# ---------------------------------------------------------------------------
# Streaming chunk
# ---------------------------------------------------------------------------


class StreamChunk(BaseModel):
    """
    A single chunk in a streaming response.

    When ``stream=True`` is passed on the :class:`~uai.UniversalAI` client,
    the ``chat()`` method returns an *iterator* of ``StreamChunk``
    objects.  Each chunk carries a delta of text content and/or tool-call
    data.  The final chunk has ``is_final=True`` and a ``finish_reason``.
    """

    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(
        default=None,
        description="Delta text content for this chunk (may be None for non-content chunks).",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Delta tool calls (partial arguments may arrive across chunks).",
    )
    finish_reason: FinishReason | None = Field(
        default=None,
        description="Populated on the final chunk.",
    )
    id: str | None = Field(
        default=None,
        description="Completion ID (usually present on the first chunk).",
    )
    model: str | None = Field(
        default=None,
        description="Model name (usually present on the first chunk).",
    )
    provider: str | None = Field(
        default=None,
        description="Provider name.",
    )
    usage: UsageMetrics | None = Field(
        default=None,
        description="Usage metrics (may appear only on the final chunk for some providers).",
    )
    ttft_ms: float | None = Field(
        default=None,
        description="Time-to-first-token in milliseconds (populated on the first chunk).",
    )
    index: int | None = Field(
        default=None,
        description="Chunk index, for multi-choice responses.",
    )
    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk in the stream.",
    )
    parsed: BaseModel | None = Field(
        default=None,
        description="Validated structured output on the final chunk (when ``output_schema`` was provided).",
    )
    raw: dict[str, Any] | None = Field(
        default=None,
        description="The original provider chunk, for debugging.",
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingResult(BaseModel):
    """A single embedding vector for one input text."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        default_factory=list,
        description="The embedding vector as a list of floats.",
    )
    dimension: int = Field(default=0, ge=0, description="Vector dimensionality.")
    index: int = Field(
        default=0, ge=0, description="Index of the input this vector corresponds to."
    )


class EmbeddingsResponse(BaseModel):
    """Normalized response for an embedding request."""

    model_config = ConfigDict(extra="forbid")

    vectors: list[EmbeddingResult] = Field(
        default_factory=list, description="Embedding vectors, one per input text."
    )
    model: str | None = Field(default=None, description="Model that served the request.")
    provider: str | None = Field(default=None, description="Provider that served the request.")
    usage: UsageMetrics = Field(
        default_factory=UsageMetrics, description="Token usage for the embedding request."
    )
    raw: dict[str, Any] | None = Field(
        default=None, description="The original provider response payload."
    )


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------


class RerankResult(BaseModel):
    """A single reranked document with its relevance score."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Index of the document in the original input list.")
    score: float = Field(description="Relevance score assigned by the reranker.")
    text: str | None = Field(
        default=None, description="The reranked document text (when returned by the provider)."
    )


class RerankResponse(BaseModel):
    """Unified response for a rerank request."""

    model_config = ConfigDict(extra="forbid")

    results: list[RerankResult] = Field(
        default_factory=list,
        description="Documents ordered by descending relevance.",
    )
    model: str | None = Field(default=None, description="Model that served the request.")
    provider: str | None = Field(default=None, description="Provider that served the request.")
    usage: UsageMetrics = Field(
        default_factory=UsageMetrics, description="Token usage for the rerank request."
    )
    raw: dict[str, Any] | None = Field(
        default=None, description="The original provider response payload."
    )
