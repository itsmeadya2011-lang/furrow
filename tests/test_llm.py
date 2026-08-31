from __future__ import annotations

import pytest

from furrow.llm import JSONParseError, extract_json


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_json():
    text = '```json\n{"passed": true, "failures": []}\n```'
    assert extract_json(text) == {"passed": True, "failures": []}


def test_extract_json_fenced_no_lang():
    text = '```\n{"x": "y"}\n```'
    assert extract_json(text) == {"x": "y"}


def test_extract_json_with_prose():
    text = 'Sure! Here is the plan:\n{"tasks": []}\nHope that helps.'
    assert extract_json(text) == {"tasks": []}


def test_extract_json_nested():
    text = 'before {"a": {"b": [1, 2]}} after'
    assert extract_json(text) == {"a": {"b": [1, 2]}}


def test_extract_json_invalid_raises():
    with pytest.raises(JSONParseError):
        extract_json("no json here at all")
