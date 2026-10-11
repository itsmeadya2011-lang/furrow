import json
import pytest

from furrow.agents.parse import extract_json


def test_extract_json_plain_object():
    s = '{"a": 1, "b": 2}'
    assert extract_json(s) == s


def test_extract_json_fenced_json():
    s = '```json\n{"a": 1}\n```'
    assert extract_json(s) == '{"a": 1}'


def test_extract_json_fenced_plain():
    s = '```\n{"a": 1, "b": 2}\n```'
    assert extract_json(s) == '{"a": 1, "b": 2}'


def test_extract_json_surrounded_by_prose():
    s = 'Here is the plan: {"x": 1, "y": 2} Hope this helps'
    assert extract_json(s) == '{"x": 1, "y": 2}'


def test_extract_json_nested_braces():
    obj = {"a": {"b": {"c": 1}, "d": [1, 2, 3]}}
    s = json.dumps(obj)
    # surrounded by text
    wrapped = f"Prose before {s} prose after"
    assert json.loads(extract_json(wrapped)) == obj


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("no json here at all")
