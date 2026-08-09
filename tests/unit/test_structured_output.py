"""Unit tests for Structured Output Parsing and Validation (Module 1.3.2).

Covers the shared parsing helpers, the client chat path (non-streaming and
streaming) populating ``parsed`` / raising ``ResponseParsingError``, the
system-prompt schema injection, cache-key partitioning across schemas, and
the opt-in retry behaviour.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

import uai.client as client_module
from uai import UniversalAI
from uai.exceptions import ResponseParsingError
from uai.middleware import RetryMiddleware
from uai.middleware.base import MiddlewareContext
from uai.structured import build_schema_prompt, extract_json_object, parse_structured_output


class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int


def good_json_response():
    """A fake httpx response whose content validates against ``Summary``."""

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            content = '{"title": "T", "key_points": ["a"], "word_count": 1}'
            return {"id": "y", "choices": [{"message": {"content": content}}]}

    return FakeResponse()


class TestStructuredHelpers:
    def test_extract_json_object_plain(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_extract_json_object_with_fences(self):
        raw = '```json\n{"a": 1}\n```'
        assert extract_json_object(raw) == {"a": 1}

    def test_extract_json_object_with_leading_prose(self):
        raw = 'Here is the result:\n{"a": 1}\nHope that helps!'
        assert extract_json_object(raw) == {"a": 1}

    def test_extract_json_object_array(self):
        assert extract_json_object("[1, 2, 3]") == [1, 2, 3]

    def test_extract_json_object_braces_inside_string(self):
        raw = '{"msg": "use {curly} braces", "n": 1}'
        assert extract_json_object(raw) == {"msg": "use {curly} braces", "n": 1}

    def test_extract_json_object_raises_when_absent(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json_object("no json here")

    def test_parse_structured_output_valid(self):
        content = '{"title": "T", "key_points": ["a"], "word_count": 1}'
        result = parse_structured_output(content, Summary, provider="deepseek")
        assert isinstance(result, Summary)
        assert result.title == "T"

    def test_parse_structured_output_bad_json(self):
        with pytest.raises(ResponseParsingError, match="could not be parsed as JSON"):
            parse_structured_output("not json", Summary, provider="deepseek")

    def test_parse_structured_output_schema_violation(self):
        content = '{"title": "T", "word_count": "many"}'
        with pytest.raises(ResponseParsingError, match="validation failed"):
            parse_structured_output(content, Summary, provider="deepseek")

    def test_build_schema_prompt_contains_schema(self):
        prompt = build_schema_prompt(Summary)
        assert "JSON Schema" in prompt
        assert '"title"' in prompt


class TestClientStructuredOutput:
    def _fake_post(self, content: str, captured: dict):
        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json

            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "id": "id-1",
                        "choices": [{"message": {"content": content}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                    }

            return FakeResponse()

        return fake_post

    def test_chat_populates_parsed(self, monkeypatch):
        captured: dict = {}
        content = '{"title": "Hello", "key_points": ["x"], "word_count": 1}'
        monkeypatch.setattr(client_module.httpx, "post", self._fake_post(content, captured))
        client = UniversalAI(api_key="k", provider="deepseek")
        result = client.chat(
            messages=[{"role": "user", "content": "Summarize"}],
            output_schema=Summary,
        )
        assert isinstance(result.parsed, Summary)
        assert result.parsed.title == "Hello"
        assert result.parsed.word_count == 1

    def test_chat_injects_schema_prompt(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            client_module.httpx,
            "post",
            self._fake_post('{"title": "T", "key_points": [], "word_count": 0}', captured),
        )
        client = UniversalAI(api_key="k", provider="deepseek")
        client.chat(
            messages=[{"role": "user", "content": "Summarize"}],
            output_schema=Summary,
        )
        messages = captured["body"]["messages"]
        assert messages[0]["role"] == "system"
        assert "JSON Schema" in messages[0]["content"]
        assert '"title"' in messages[0]["content"]
        # Original messages still present after the injected system message.
        assert messages[-1] == {"role": "user", "content": "Summarize"}

    def test_chat_no_schema_means_no_injection(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            client_module.httpx,
            "post",
            self._fake_post("plain text", captured),
        )
        client = UniversalAI(api_key="k", provider="deepseek")
        client.chat(messages=[{"role": "user", "content": "Hi"}])
        assert captured["body"]["messages"][0]["role"] == "user"

    def test_chat_raises_on_bad_json(self, monkeypatch):
        monkeypatch.setattr(client_module.httpx, "post", self._fake_post("not json at all", {}))
        client = UniversalAI(api_key="k", provider="deepseek")
        with pytest.raises(ResponseParsingError, match="could not be parsed as JSON"):
            client.chat(
                messages=[{"role": "user", "content": "Summarize"}],
                output_schema=Summary,
            )

    def test_chat_raises_on_schema_violation(self, monkeypatch):
        monkeypatch.setattr(
            client_module.httpx,
            "post",
            self._fake_post('{"title": 42, "key_points": "nope", "word_count": "x"}', {}),
        )
        client = UniversalAI(api_key="k", provider="deepseek")
        with pytest.raises(ResponseParsingError, match="validation failed"):
            client.chat(
                messages=[{"role": "user", "content": "Summarize"}],
                output_schema=Summary,
            )


class TestStreamingStructuredOutput:
    def _chunk(self, content: str | None, finish: bool = False) -> str:
        data = {
            "id": "id-1",
            "choices": [
                {
                    "delta": {"content": content} if content is not None else {},
                    "finish_reason": "stop" if finish else None,
                }
            ],
        }
        return f"data: {json.dumps(data)}"

    def _stream_lines(self, parts: list[str]):
        lines = [self._chunk(p) for p in parts]
        lines.append(self._chunk(None, finish=True))
        lines.append("data: [DONE]")
        return lines

    def _fake_stream(self, lines: list[str], captured: dict):
        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                return iter(lines)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_stream(method, url, headers=None, json=None, timeout=None):
            captured["body"] = json
            return FakeResponse()

        return fake_stream

    def test_stream_populates_parsed_on_final_chunk(self, monkeypatch):
        captured: dict = {}
        lines = self._stream_lines(
            ['{"title": "Stream', 'ed", "key_points": ["a"], "word_count": 2}']
        )
        monkeypatch.setattr(client_module.httpx, "stream", self._fake_stream(lines, captured))
        client = UniversalAI(api_key="k", provider="deepseek")
        chunks = list(
            client.chat(
                messages=[{"role": "user", "content": "Summarize"}],
                output_schema=Summary,
                stream=True,
            )
        )
        parsed = [c.parsed for c in chunks if c.parsed is not None]
        assert len(parsed) == 1
        assert isinstance(parsed[0], Summary)
        assert parsed[0].title == "Streamed"
        # Schema prompt is injected into the streaming request too.
        assert captured["body"]["messages"][0]["role"] == "system"

    def test_stream_raises_on_bad_json(self, monkeypatch):
        captured: dict = {}
        lines = self._stream_lines(["this is not json"])
        monkeypatch.setattr(client_module.httpx, "stream", self._fake_stream(lines, captured))
        client = UniversalAI(api_key="k", provider="deepseek")
        with pytest.raises(ResponseParsingError, match="could not be parsed as JSON"):
            list(
                client.chat(
                    messages=[{"role": "user", "content": "Summarize"}],
                    output_schema=Summary,
                    stream=True,
                )
            )

    def test_stream_validates_when_stream_ends_without_done(self, monkeypatch):
        # Provider closes the stream with no [DONE] and no finish_reason.
        content = '{"title": "T", "key_points": ["a"], "word_count": 1}'
        lines = [self._chunk(content[:10]), self._chunk(content[10:])]

        class FakeResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_lines(self):
                return iter(lines)

        def fake_stream(method, url, headers=None, json=None, timeout=None):
            return FakeResponse()

        monkeypatch.setattr(client_module.httpx, "stream", fake_stream)
        client = UniversalAI(api_key="k", provider="deepseek")
        chunks = list(
            client.chat(
                messages=[{"role": "user", "content": "Summarize"}],
                output_schema=Summary,
                stream=True,
            )
        )
        parsed = [c.parsed for c in chunks if c.parsed is not None]
        assert len(parsed) == 1
        assert parsed[0].title == "T"

    def test_cache_does_not_reuse_parsed_across_schemas(self, monkeypatch):
        from uai.middleware import CacheMiddleware

        class OtherSchema(BaseModel):
            # Same payload is valid for both schemas; only the schema class
            # differs, so the cache key must distinguish them.
            title: str

        calls = {"n": 0}

        def counting_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            return good_json_response()

        monkeypatch.setattr(client_module.httpx, "post", counting_post)
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(CacheMiddleware(ttl=60))
        messages = [{"role": "user", "content": "Summarize"}]

        r1 = client.chat(messages=messages, output_schema=Summary)
        assert isinstance(r1.parsed, Summary)
        # Same messages, different schema -> must hit the network again.
        r2 = client.chat(messages=messages, output_schema=OtherSchema)
        assert isinstance(r2.parsed, OtherSchema)
        assert calls["n"] == 2
        # Identical schema -> cache hit.
        r3 = client.chat(messages=messages, output_schema=Summary)
        assert isinstance(r3.parsed, Summary)
        assert calls["n"] == 2


class TestRetryOnParsingError:
    def test_default_does_not_retry_parsing_error(self):
        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            raise ResponseParsingError("bad json")

        mw = RetryMiddleware(max_retries=3, base_delay=0.001, jitter=False)
        ctx = MiddlewareContext(operation="chat", request=None)
        with pytest.raises(ResponseParsingError):
            mw.execute(call_next, ctx)
        assert calls["n"] == 1

    def test_opt_in_retries_parsing_error(self):
        calls = {"n": 0}

        def call_next():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ResponseParsingError("bad json")
            return "ok"

        mw = RetryMiddleware(
            max_retries=3, base_delay=0.001, jitter=False, retry_on_parsing_error=True
        )
        ctx = MiddlewareContext(operation="chat", request=None)
        assert mw.execute(call_next, ctx) == "ok"
        assert calls["n"] == 3

    def test_client_retries_parsing_error_when_enabled(self, monkeypatch):
        calls = {"n": 0}

        def flaky_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 2:
                return self._bad_json_response()
            return good_json_response()

        monkeypatch.setattr(client_module.httpx, "post", flaky_post)
        client = UniversalAI(api_key="k", provider="deepseek")
        client.use(
            RetryMiddleware(
                max_retries=3, base_delay=0.001, jitter=False, retry_on_parsing_error=True
            )
        )
        result = client.chat(
            messages=[{"role": "user", "content": "Summarize"}],
            output_schema=Summary,
        )
        assert isinstance(result.parsed, Summary)
        assert calls["n"] == 2

    @staticmethod
    def _bad_json_response():
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"id": "x", "choices": [{"message": {"content": "not json"}}]}

        return FakeResponse()
