from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from furrow.jsonutil import (
    JSONExtractionError,
    extract_json,
    parse_model_json,
    strip_code_fences,
    truncate,
)


class TaskModel(BaseModel):
    id: str
    description: str


class PlanModel(BaseModel):
    tasks: list[TaskModel]
    rationale: str | None = None


def test_fenced_object_is_unwrapped():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_fenced_array_with_uppercase_info_string_is_unwrapped():
    assert strip_code_fences('```JSON\n[1, 2]\n```') == "[1, 2]"


def test_bare_fence_is_unwrapped():
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_unterminated_fence_is_unwrapped():
    assert strip_code_fences('```json\n{"a": 1}') == '{"a": 1}'


def test_fence_without_trailing_newline_is_unwrapped():
    assert strip_code_fences('```json{"a": 1}\n```') == '{"a": 1}'


def test_text_without_fence_is_unchanged():
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'
    assert strip_code_fences("Here is the plan:\n{}") == "Here is the plan:\n{}"
    assert strip_code_fences("") == ""


def test_backticks_inside_a_string_are_not_mangled():
    payload = '{"note": "run ```pytest``` first"}'
    assert strip_code_fences(payload) == payload


def test_trailing_backticks_in_prose_are_not_mangled():
    assert strip_code_fences("use ``` here") == "use ``` here"


def test_non_json_fence_content_is_still_unwrapped():
    assert strip_code_fences("```python\nprint(1)\n```") == "print(1)"


def test_fence_indented_by_whitespace_is_unwrapped():
    assert strip_code_fences('  \n```json\n{"a": 1}\n```') == '{"a": 1}'


def test_leading_fence_only_is_unwrapped():
    # The fence in the middle belongs to the prose and has to survive; only the fence
    # that opens the text and the one that closes the very end are removed.
    text = '```json\n{"a": 1}\n```\nAnd here is the same object: ```json\n{"a": 1}\n```'
    assert strip_code_fences(text) == '{"a": 1}\n```\nAnd here is the same object: ```json\n{"a": 1}'
    assert extract_json(text) == '{"a": 1}'


def test_fence_followed_by_trailing_newline_is_unwrapped():
    assert strip_code_fences('```json\n{"a": 1}\n```\n') == '{"a": 1}'


def test_extract_json_pure_object():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_pure_array():
    assert extract_json("[1, 2, 3]") == "[1, 2, 3]"


def test_extract_json_with_leading_prose():
    text = 'Here is the plan:\n{"tasks": []}'
    assert extract_json(text) == '{"tasks": []}'


def test_extract_json_with_trailing_prose():
    text = '{"tasks": []}\n\nLet me know if you want changes.'
    assert extract_json(text) == '{"tasks": []}'


def test_extract_json_with_prose_on_both_sides_and_a_fence():
    text = 'Sure!\n```json\n{"tasks": []}\n```\nHope that helps.'
    assert extract_json(text) == '{"tasks": []}'


def test_extract_json_handles_nested_braces():
    text = '{"outer": {"inner": {"deep": 1}}, "tail": true}'
    assert extract_json(text) == text
    assert json.loads(extract_json(text))["outer"]["inner"]["deep"] == 1


def test_extract_json_handles_nested_arrays_inside_objects():
    text = '{"tasks": [{"id": "1", "tags": ["a", "b"]}]}'
    assert extract_json(text) == text


def test_extract_json_ignores_braces_inside_strings():
    text = '{"summary": "use { and } carefully"}'
    assert extract_json(text) == text
    assert json.loads(extract_json(text))["summary"] == "use { and } carefully"


def test_extract_json_handles_escaped_quotes_inside_strings():
    text = r'{"summary": "he said \"stop {\" loudly", "ok": true}'
    assert extract_json(text) == text
    assert json.loads(extract_json(text))["summary"] == 'he said "stop {" loudly'


def test_extract_json_handles_escaped_backslash_before_quote():
    text = r'{"path": "C:\\", "note": "trailing backslash then quote:"}'
    assert extract_json(text) == text
    assert json.loads(extract_json(text))["path"] == "C:\\"


def test_extract_json_tolerates_single_quoted_strings():
    assert extract_json("{'a': 'b}'}") == "{'a': 'b}'}"


def test_extract_json_returns_first_of_several_values():
    text = '{"example": true}\n\nAnd the real one: {"real": true}'
    assert extract_json(text) == '{"example": true}'


def test_extract_json_first_of_several_values_ignores_trailing_json():
    assert extract_json('{"a": 1} {"b": 2}') == '{"a": 1}'


def test_extract_json_skips_candidates_that_never_balance():
    text = 'Stray { brace and a { here: {"a": 1}'
    assert extract_json(text) == '{"a": 1}'


def test_extract_json_takes_the_first_balanced_candidate_even_if_unparseable():
    # Documented tradeoff: prose is full of balanced braces, and taking the first one
    # means this placeholder shadows the real payload. parse_model_json turns that into a
    # retryable JSONExtractionError rather than a crash, which is the better trade than
    # returning a fragment of a deeply nested value that happens to parse.
    text = "Set {env} first, then run the plan: {\"a\": 1}"
    assert extract_json(text) == "{env}"
    with pytest.raises(JSONExtractionError):
        parse_model_json(text, TaskModel)


def test_extract_json_accepts_deeply_nested_input():
    depth = 2000
    text = "[" * depth + "1" + "]" * depth
    assert extract_json(text) == text


def test_parse_model_json_reports_deep_nesting_as_retryable_not_a_crash():
    depth = 2000
    text = "[" * depth + "1" + "]" * depth
    with pytest.raises(JSONExtractionError):
        parse_model_json(text, TaskModel)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n\t ",
        "no json here at all",
        "{",
        "[",
        '{"a": 1',
        '{"tasks": [{"id": "1", "description": "do',
        '[1, 2',
        '{"a": "unterminated string}',
        "{'a': 'b",
        "```json\n",
        "```json\n```",
        "```",
    ],
)
def test_extract_json_raises_json_extraction_error(text):
    with pytest.raises(JSONExtractionError):
        extract_json(text)


@pytest.mark.parametrize("text", ["42", '"a bare scalar"'])
def test_extract_json_rejects_scalars(text):
    # Only objects and arrays are recoverable: a prompt that demands a schema gets an
    # object back, and anything else is a signal that the model ignored it, which the
    # caller should treat as retryable rather than guess at.
    with pytest.raises(JSONExtractionError):
        extract_json(text)


def test_extract_json_handles_crlf_fenced_payload():
    assert extract_json('```json\r\n{"a": 1}\r\n```') == '{"a": 1}'


def test_extract_json_handles_non_ascii_values():
    text = '{"summary": "café \U0001f600"}'
    assert json.loads(extract_json(text))["summary"] == "café \U0001f600"


def test_extract_json_does_not_raise_json_decode_error():
    # Callers branch on the type, so these two must never overlap.
    assert not issubclass(JSONExtractionError, json.JSONDecodeError)
    with pytest.raises(JSONExtractionError) as info:
        extract_json('{"a": 1')
    assert not isinstance(info.value, json.JSONDecodeError)


def test_parse_model_json_plain_payload():
    task = parse_model_json('{"id":"1","description":"do"}', TaskModel)
    assert task == TaskModel(id="1", description="do")


def test_parse_model_json_with_fence_and_prose():
    text = 'Here you go:\n```json\n{"id": "2", "description": "fix"}\n```\nDone.'
    task = parse_model_json(text, TaskModel)
    assert task.description == "fix"


def test_parse_model_json_nested_model():
    text = 'Plan: {"tasks": [{"id": "1", "description": "do"}], "rationale": "because"}'
    plan = parse_model_json(text, PlanModel)
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "because"


def test_parse_model_json_raises_extraction_error_for_prose_only():
    with pytest.raises(JSONExtractionError):
        parse_model_json("I am afraid I cannot do that", TaskModel)


def test_parse_model_json_raises_extraction_error_for_truncated_output():
    with pytest.raises(JSONExtractionError):
        parse_model_json('{"id": "1", "description": "do someth', TaskModel)


def test_parse_model_json_raises_extraction_error_not_decode_error():
    with pytest.raises(JSONExtractionError) as info:
        parse_model_json('{"id": "1", "description": "do someth', TaskModel)
    assert not isinstance(info.value, json.JSONDecodeError)


def test_parse_model_json_raises_validation_error_on_schema_mismatch():
    # A well-formed payload of the wrong shape is permanent, not retryable, so it must
    # surface as ValidationError and never be swallowed as an extraction failure.
    with pytest.raises(ValidationError) as info:
        parse_model_json('{"identifier": "1", "text": "do"}', TaskModel)
    assert not isinstance(info.value, JSONExtractionError)


def test_parse_model_json_missing_required_field_is_a_validation_error():
    with pytest.raises(ValidationError):
        parse_model_json('{"id": "1"}', TaskModel)


def test_truncate_leaves_short_text_alone():
    assert truncate("hello", 10) == "hello"
    assert truncate("hello") == "hello"
    assert truncate("") == ""


def test_truncate_leaves_exact_length_alone():
    assert truncate("12345", 5) == "12345"


def test_truncate_marks_the_elision():
    text = "x" * 50
    result = truncate(text, 10)
    assert result.startswith("x" * 10)
    assert "[truncated 40 characters]" in result
    assert result.endswith("...")


def test_truncate_defaults_to_four_thousand_characters():
    text = "y" * 5000
    result = truncate(text)
    assert result.startswith("y" * 4000)
    assert "[truncated 1000 characters]" in result


@pytest.mark.parametrize("limit", [0, -1, -1000])
def test_truncate_rejects_non_positive_limits(limit):
    with pytest.raises(ValueError):
        truncate("some text", limit)
