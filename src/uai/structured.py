"""
Structured Output Parsing and Validation (Module 1.3.2).

Shared helpers used by both the client chat path and the provider
adapters to:

* build a system-prompt instruction from a Pydantic model's JSON schema
  (injected into the request so the model is nudged toward valid JSON),
* parse and validate a model's string payload against the schema, and
* raise :class:`~uai.exceptions.ResponseParsingError` when the payload is
  malformed JSON or violates the schema — an error designed to be caught
  (and optionally retried) by the middleware pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from uai.exceptions import ResponseParsingError

# Common LLM wrapping artifacts that break naive json.loads().
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json_object(content: str) -> Any:
    """
    Parse *content* as JSON, tolerating LLM wrapping artifacts.

    Strips markdown code fences and leading prose before attempting to
    parse, so a model that wraps its answer in ```json ... ``` still
    validates.

    :param content: The raw string payload from the model.
    :raises json.JSONDecodeError: If no valid JSON can be extracted.
    """
    cleaned = _CODE_FENCE.sub("", content).strip()

    # Try the whole (cleaned) string first — the common case.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the first {...} or [...] block in the payload.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            char = cleaned[i]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start : i + 1])
    raise json.JSONDecodeError("No JSON object found in content", content, 0) from None


def build_schema_prompt(schema: type[BaseModel]) -> str:
    """
    Build a system-prompt instruction that asks the model to return a JSON
    object matching *schema*'s JSON Schema.

    :param schema: A Pydantic model class.
    :return: A prompt string suitable for a system message.
    """
    schema_json = json.dumps(schema.model_json_schema())
    return (
        "You must respond with a single JSON object that conforms exactly "
        f"to the following JSON Schema. Do not wrap it in code fences and "
        f"do not include any text outside the JSON object.\n\nSchema:\n{schema_json}"
    )


def parse_structured_output(
    content: str,
    schema: type[BaseModel],
    *,
    provider: str | None = None,
) -> BaseModel:
    """
    Parse *content* and validate it against *schema*.

    :param content: The string payload returned by the model.
    :param schema: The Pydantic model to validate against.
    :param provider: Optional provider name attached to errors.
    :raises ResponseParsingError: If the payload is not valid JSON or does
        not conform to the schema.
    :return: An instance of *schema*.
    """
    try:
        payload = extract_json_object(content)
    except json.JSONDecodeError as exc:
        raise ResponseParsingError(
            "Structured output could not be parsed as JSON", provider=provider
        ) from exc

    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise ResponseParsingError(
            f"Structured output validation failed: {exc}", provider=provider
        ) from exc


__all__ = [
    "build_schema_prompt",
    "extract_json_object",
    "parse_structured_output",
]
