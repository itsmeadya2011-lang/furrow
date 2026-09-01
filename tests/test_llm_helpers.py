from furrow.llm_helpers import extract_json


def test_extract_json_plain():
    result = extract_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_extract_json_markdown_fence():
    text = 'Here is the JSON:\n```json\n{"foo": 42}\n```\nDone.'
    result = extract_json(text)
    assert result == {"foo": 42}


def test_extract_json_embedded_in_prose():
    text = 'Sure! Here you go: {"a": 1, "b": [2, 3]} — that should work.'
    result = extract_json(text)
    assert result == {"a": 1, "b": [2, 3]}


def test_extract_json_empty_string():
    result = extract_json("")
    assert result is None
