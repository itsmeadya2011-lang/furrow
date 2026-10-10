from furrow.llm import extract_json


def test_extract_json_plain_stripped():
    assert extract_json('  {"a": 1}  ') == '{"a": 1}'


def test_extract_json_fenced_json_lang():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_fenced_no_lang():
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_prose_around():
    text = 'Sure! Here is the plan:\n{"a": 1}\nHope that helps.'
    assert extract_json(text) == '{"a": 1}'


def test_extract_json_nested_braces():
    text = '{"a": {"b": {"c": 1}}}'
    assert extract_json(text) == text


def test_extract_json_no_braces_returns_stripped():
    assert extract_json("  hello world  ") == "hello world"


def test_extract_json_empty_string():
    assert extract_json("") == ""