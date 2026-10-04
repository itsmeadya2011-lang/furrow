from __future__ import annotations

import pytest

from furrow.agents.jsonparse import extract_json


def test_plain_object():
    text = '{"a": 1, "b": [1, 2, 3]}'
    assert extract_json(text) == text


def test_plain_array():
    text = '[{"id": "1"}, {"id": "2"}]'
    assert extract_json(text) == text


def test_fenced_json_lang_tag():
    text = '```json\n{"a": 1}\n```'
    assert extract_json(text) == '{"a": 1}'


def test_fenced_json_plain():
    text = '```\n{"a": 1}\n```'
    assert extract_json(text) == '{"a": 1}'


def test_leading_and_trailing_prose():
    text = 'Here is the plan:\n```json\n{"tasks": []}\n```\nLet me know if that works.'
    assert extract_json(text) == '{"tasks": []}'


def test_prose_without_fence():
    text = 'Sure! Here is the result: {"answer": 42} Hope that helps.'
    assert extract_json(text) == '{"answer": 42}'


def test_braces_inside_strings():
    text = '{"code": "if (x) { return y; }", "ok": true}'
    assert extract_json(text) == text


def test_braces_inside_strings_with_array():
    text = '{"items": ["{a}", "[b]"], "n": 2}'
    assert extract_json(text) == text


def test_empty_string_raises():
    with pytest.raises(ValueError):
        extract_json("")


def test_whitespace_only_raises():
    with pytest.raises(ValueError):
        extract_json("   \n  ")


def test_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("This is just prose with no JSON at all.")


def test_array_with_braces_in_strings():
    text = 'Here is the list: ["{not json}", "real"] done.'
    assert extract_json(text) == '["{not json}", "real"]'