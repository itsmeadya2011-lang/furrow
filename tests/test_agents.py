import json

import pytest

from furrow.agents.prompts import extract_json


def test_extract_json_bare_object():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_fenced():
    text = '```json\n{"a": 1}\n```'
    assert json.loads(extract_json(text)) == {"a": 1}


def test_extract_json_fenced_no_language():
    text = '```\n{"a": 1}\n```'
    assert json.loads(extract_json(text)) == {"a": 1}


def test_extract_json_with_preamble():
    text = 'Sure! Here is the plan:\n{"tasks": [], "rationale": "ok"}'
    assert json.loads(extract_json(text)) == {"tasks": [], "rationale": "ok"}


def test_extract_json_trailing_prose_after_object():
    text = '{"a": 1}\n\nNote: this is prose with {braces}'
    assert json.loads(extract_json(text)) == {"a": 1}


def test_extract_json_nested_braces_and_escapes():
    text = '{"cmd": "echo \\"{\\"a\\": 1}\\"", "n": {"inner": true}}'
    assert json.loads(extract_json(text))["n"] == {"inner": True}


def test_extract_json_array_of_objects_returns_first_object():
    text = '[{"a": 1}, {"b": 2}]'
    assert json.loads(extract_json(text)) == {"a": 1}


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json("no json here")


def test_extract_json_unterminated_raises():
    with pytest.raises(ValueError, match="Unterminated JSON object"):
        extract_json('{"a": 1')


def test_extract_json_whitespace_around():
    text = '\n\n  {"a": 1}  \n\n'
    assert json.loads(extract_json(text)) == {"a": 1}
