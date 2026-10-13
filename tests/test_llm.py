from __future__ import annotations

import json

import pytest

from furrow.llm import _find_json_object, _strip_code_fences, extract_json
from tests.fakes import FakeLLMClient


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}

    def test_json_in_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_json_in_plain_fence(self):
        text = '```\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_json_in_python_fence(self):
        text = '```python\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_prose_with_embedded_json(self):
        text = 'Here is the result:\n{"a": 1}\nThanks for reading.'
        assert extract_json(text) == {"a": 1}

    def test_prose_returns_first_json_object(self):
        text = 'Result: {"a": 1} and more {"b": 2}'
        assert extract_json(text) == {"a": 1}

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, {"x": true}]}}'
        assert extract_json(text) == {"outer": {"inner": [1, 2, {"x": True}]}}

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("")

    def test_no_braces_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("just some prose without any json at all")

    def test_json_array_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("[1, 2, 3]")

    def test_unclosed_object_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("{not valid json")

    def test_malformed_json_with_braces_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json('{"a": invalid}')

    def test_string_value_with_braces(self):
        text = '{"msg": "a { b } c"}'
        assert extract_json(text) == {"msg": "a { b } c"}

    def test_truncates_long_context_in_error(self):
        text = "no json here " * 100  # > 500 chars
        with pytest.raises(ValueError) as exc_info:
            extract_json(text)
        msg = str(exc_info.value)
        assert text[:500] in msg
        assert text[600:] not in msg


class TestStripCodeFences:
    def test_strips_json_fence(self):
        out = _strip_code_fences('```json\n{"a":1}\n```')
        assert "```" not in out
        assert json.loads(out) == {"a": 1}

    def test_strips_plain_fence(self):
        out = _strip_code_fences('```\n{"a":1}\n```')
        assert "```" not in out
        assert json.loads(out) == {"a": 1}

    def test_no_fence_returns_text(self):
        assert _strip_code_fences('{"a":1}') == '{"a":1}'

    def test_unclosed_fence_returns_remainder(self):
        out = _strip_code_fences('```json\n{"a": 1}')
        assert "```" not in out
        assert json.loads(out) == {"a": 1}


class TestFindJsonObject:
    def test_returns_none_when_no_brace(self):
        assert _find_json_object("no braces here") is None

    def test_finds_first_balanced_object(self):
        assert _find_json_object('x {"a":1} y {"b":2}') == '{"a":1}'

    def test_returns_none_when_unbalanced(self):
        assert _find_json_object('{"a":1') is None


class TestFakeLLMClient:
    async def test_complete_json_with_dict(self):
        fake = FakeLLMClient(responses=[{"tasks": [], "rationale": "done"}])
        data = await fake.complete_json("some prompt")
        assert data == {"tasks": [], "rationale": "done"}

    async def test_complete_json_with_json_string(self):
        fake = FakeLLMClient(responses=['```json\n{"a": 2}\n```'])
        data = await fake.complete_json("some prompt")
        assert data == {"a": 2}

    async def test_complete_json_raises_on_invalid_json(self):
        fake = FakeLLMClient(responses=["totally not json"])
        with pytest.raises(ValueError):
            await fake.complete_json("some prompt")

    async def test_complete_json_raises_on_scripted_exception(self):
        fake = FakeLLMClient(responses=[ValueError("boom")])
        with pytest.raises(ValueError, match="boom"):
            await fake.complete_json("some prompt")

    async def test_complete_returns_string(self):
        fake = FakeLLMClient(responses=["hello world"])
        out = await fake.complete("prompt")
        assert out == "hello world"

    async def test_complete_messages_returns_string(self):
        fake = FakeLLMClient(responses=["assistant reply"])
        out = await fake.complete_messages([{"role": "user", "content": "hi"}])
        assert out == "assistant reply"

    async def test_calls_recorded_in_order(self):
        fake = FakeLLMClient(responses=[{"a": 1}, "reply", "second"])
        await fake.complete_json("p1")
        await fake.complete_messages([{"role": "user", "content": "hi"}])
        await fake.complete("p2")
        methods = [c[0] for c in fake.calls]
        assert methods == ["complete_json", "complete_messages", "complete"]

    async def test_records_prompts(self):
        fake = FakeLLMClient(responses=[{"a": 1}])
        await fake.complete_json("the goal prompt")
        assert fake.calls[0][1] == "the goal prompt"

    async def test_complete_messages_consumes_queue_sequentially(self):
        fake = FakeLLMClient(responses=["first", "second", "third"])
        assert await fake.complete_messages([{"role": "user", "content": "a"}]) == "first"
        assert await fake.complete_messages([{"role": "user", "content": "b"}]) == "second"
        assert await fake.complete_messages([{"role": "user", "content": "c"}]) == "third"
