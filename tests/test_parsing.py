"""Tests for the JSON extraction helpers used to talk to sloppy LLMs."""

from __future__ import annotations

import pytest

from furrow.agents.parsing import (
    LLMParseError,
    extract_json_array,
    extract_json_object,
    find_json_bool,
    strip_code_fences,
)


def test_strip_code_fences_removes_a_json_fence() -> None:
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_removes_a_bare_fence() -> None:
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_leaves_plain_text_alone() -> None:
    assert strip_code_fences("  just prose  ") == "just prose"


def test_extract_json_object_from_a_fenced_block() -> None:
    payload = '```json\n{"tasks": [], "rationale": "ok"}\n```'
    assert extract_json_object(payload) == {"tasks": [], "rationale": "ok"}


def test_extract_json_object_ignores_trailing_prose() -> None:
    payload = '{"passed": true}\n\nThat is the plan. Hope it helps!'
    assert extract_json_object(payload) == {"passed": True}


def test_extract_json_object_ignores_leading_prose() -> None:
    payload = 'Here is the plan:\n{"passed": false}'
    assert extract_json_object(payload) == {"passed": False}


def test_extract_json_object_handles_braces_inside_strings() -> None:
    payload = '{"summary": "wrap {this} in braces", "count": 2}'
    assert extract_json_object(payload) == {"summary": "wrap {this} in braces", "count": 2}


def test_extract_json_object_handles_escaped_quotes() -> None:
    payload = '{"summary": "say \\"hi\\" then {brace}", "n": 1}'
    assert extract_json_object(payload) == {"summary": 'say "hi" then {brace}', "n": 1}


def test_extract_json_object_raises_on_invalid_json() -> None:
    with pytest.raises(LLMParseError) as excinfo:
        extract_json_object('{"a": }')
    message = str(excinfo.value)
    assert "invalid JSON" in message
    assert '{"a": }' in message
    assert excinfo.value.raw == '{"a": }'


def test_extract_json_object_raises_when_there_is_no_object() -> None:
    with pytest.raises(LLMParseError) as excinfo:
        extract_json_object("there is no json here")
    assert "no JSON object found" in str(excinfo.value)
    assert excinfo.value.raw == "there is no json here"


def test_extract_json_array_returns_a_list() -> None:
    assert extract_json_array('```json\n["a", "b"]\n```') == ["a", "b"]


def test_extract_json_array_handles_trailing_prose() -> None:
    assert extract_json_array('["a"]\n\nThat is the list.') == ["a"]


def test_extract_json_array_raises_on_invalid_json() -> None:
    with pytest.raises(LLMParseError) as excinfo:
        extract_json_array("[1, 2")
    assert "no JSON array found" in str(excinfo.value)


def test_extract_json_array_rejects_a_json_object() -> None:
    with pytest.raises(LLMParseError) as excinfo:
        extract_json_array('{"a": 1}')
    assert "no JSON array found" in str(excinfo.value)


def test_find_json_bool_reads_true_and_false() -> None:
    assert find_json_bool('{"passed": true}', "passed") is True
    assert find_json_bool('{"passed": false}', "passed") is False
    assert find_json_bool("{'passed': TRUE}", "passed") is True


def test_find_json_bool_returns_none_when_the_key_is_absent() -> None:
    assert find_json_bool('{"status": "ok"}', "passed") is None


def test_find_json_bool_returns_none_for_prose_that_mentions_passing() -> None:
    assert find_json_bool("All tests passed", "passed") is None


def test_llm_parse_error_is_a_value_error() -> None:
    error = LLMParseError("raw text", "boom")
    assert isinstance(error, ValueError)
    assert error.raw == "raw text"
    assert str(error) == "boom"
