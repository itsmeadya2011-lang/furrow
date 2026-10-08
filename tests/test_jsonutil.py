import json

import pytest

from furrow.agents.jsonutil import parse_json, strip_code_fences


def test_strip_code_fences_plain():
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_strip_code_fences_with_json_fence():
    text = '```json\n{"a": 1}\n```'
    assert strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_multiline_fence():
    text = '```json\n{"a": 1, "b": [2, 3]}\n```'
    assert strip_code_fences(text) == '{"a": 1, "b": [2, 3]}'


def test_parse_json_with_fence():
    text = '```json\n{"tasks": [], "rationale": "ok"}\n```'
    assert parse_json(text) == {"tasks": [], "rationale": "ok"}


def test_parse_json_with_prose():
    text = 'Sure! Here it is: {"a": 1} hope that helps'
    assert parse_json(text) == {"a": 1}


def test_parse_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json("not json at all")