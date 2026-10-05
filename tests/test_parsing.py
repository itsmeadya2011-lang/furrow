"""Tests for :func:`furrow.agents.parsing.extract_json`.

Every expected value below is derived by inspecting the documented contract of
``extract_json`` (fence bodies, then the stripped response, then balanced
``{...}``/``[...]`` spans, preferring a value that carries content).
"""

import pytest

from furrow.agents.parsing import extract_json


def test_extract_json_bare_object_with_surrounding_whitespace() -> None:
    assert extract_json('  \n {"rationale": "ok"} \n\t ') == {"rationale": "ok"}


def test_extract_json_fenced_json_block() -> None:
    response = '```json\n{"tasks": [{"id": "1"}]}\n```'
    assert extract_json(response) == {"tasks": [{"id": "1"}]}


def test_extract_json_bare_fence_without_language_tag() -> None:
    response = '```\n{"tasks": [{"id": "1"}]}\n```'
    assert extract_json(response) == {"tasks": [{"id": "1"}]}


def test_extract_json_fence_language_tag_is_case_insensitive() -> None:
    response = '```JSON\n{"rationale": "ok"}\n```'
    assert extract_json(response) == {"rationale": "ok"}


def test_extract_json_prefers_fence_body_over_prose() -> None:
    response = 'Sure!\n```json\n{"rationale": "ok"}\n```\nAnything else?'
    assert extract_json(response) == {"rationale": "ok"}


def test_extract_json_skips_empty_fence_in_favour_of_a_later_one() -> None:
    response = '```json\n{}\n```\n```json\n{"rationale": "ok"}\n```'
    assert extract_json(response) == {"rationale": "ok"}


def test_extract_json_prose_before_and_after_the_document() -> None:
    response = 'Here is the plan:\n{"rationale": "ok"}\nLet me know.'
    assert extract_json(response) == {"rationale": "ok"}


def test_extract_json_bracket_inside_string_value() -> None:
    response = '{"summary": "use { and } carefully"}'
    assert extract_json(response)["summary"] == "use { and } carefully"


def test_extract_json_bracket_inside_string_value_after_prose() -> None:
    response = 'Result: {"summary": "use { and } carefully"}'
    assert extract_json(response)["summary"] == "use { and } carefully"


def test_extract_json_escaped_quote_inside_string_value() -> None:
    response = '{"summary": "say \\"hi\\""}'
    assert extract_json(response)["summary"] == 'say "hi"'


def test_extract_json_escaped_quote_inside_string_value_after_prose() -> None:
    response = 'Result: {"summary": "say \\"hi\\""}'
    assert extract_json(response)["summary"] == 'say "hi"'


def test_extract_json_prose_mentioning_empty_object_loses_to_payload() -> None:
    response = 'Use {} as the default.\n{"rationale": "ok"}'
    assert extract_json(response) == {"rationale": "ok"}


def test_extract_json_empty_object_is_the_last_resort() -> None:
    result = extract_json("{}")
    assert result == {}
    assert isinstance(result, dict)


def test_extract_json_empty_list_is_the_last_resort() -> None:
    result = extract_json("[]")
    assert result == []
    assert isinstance(result, list)


def test_extract_json_bare_null_is_the_last_resort() -> None:
    assert extract_json("null") is None


def test_extract_json_top_level_list() -> None:
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_top_level_list_after_prose() -> None:
    assert extract_json('Items: [1, 2, 3]') == [1, 2, 3]


def test_extract_json_raises_value_error_without_any_json() -> None:
    with pytest.raises(ValueError):
        extract_json("I could not comply.")