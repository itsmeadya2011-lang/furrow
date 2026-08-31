import pytest

from furrow.jsonutils import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_json_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_plain_fence():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_prose_surrounding():
    assert extract_json("Here is the plan: {\"x\": [1, 2]} thanks") == {"x": [1, 2]}


def test_array_parsing():
    assert extract_json("plain [1, 2, 3]") == [1, 2, 3]


def test_trailing_garbage_trimmed():
    assert extract_json('{"a": 1} and then some text') == {"a": 1}


def test_default_returned_when_no_json():
    assert extract_json("no json here", default={"z": 0}) == {"z": 0}


def test_raises_when_no_json_and_no_default():
    with pytest.raises(ValueError):
        extract_json("nothing to see")


def test_nested_object():
    text = '```json\n{"tasks": [{"id": "1", "files": ["a.py"]}]}\n```'
    assert extract_json(text) == {"tasks": [{"id": "1", "files": ["a.py"]}]}
