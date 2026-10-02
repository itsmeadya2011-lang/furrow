"""Regression tests for ``furrow.core.json_utils.extract_json``.

The parser has to survive whatever a model actually emits: fenced blocks, prose
around the payload, braces and quotes inside string values.  When it cannot find
a payload it must raise ``ValueError`` quoting the offending text, because that
text is the only diagnostic the caller has.
"""

from __future__ import annotations

import pytest

from furrow.core.json_utils import extract_json

# ------------------------------------------------------------------ bare / fenced


def test_bare_json_object_parses():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_fenced_json_block_with_language_tag_parses():
    text = '```json\n{"passed": true, "summary": "ok"}\n```'
    assert extract_json(text) == {"passed": True, "summary": "ok"}


def test_fenced_json_block_without_language_tag_parses():
    text = '```\n{"tasks": [{"id": "1"}]}\n```'
    assert extract_json(text) == {"tasks": [{"id": "1"}]}


def test_fenced_block_surrounded_by_prose_parses():
    text = 'Here is the result you asked for:\n```json\n{"ok": true}\n```\nHope that helps.'
    assert extract_json(text) == {"ok": True}


def test_fenced_block_with_trailing_whitespace_parses():
    text = '```json\n{"n": 2}\n```   \n\n'
    assert extract_json(text) == {"n": 2}


# --------------------------------------------------------------- embedded in prose


def test_json_embedded_in_surrounding_prose_parses():
    text = 'Here is the plan: {"tasks": [{"id": "1"}]} hope that helps'
    assert extract_json(text) == {"tasks": [{"id": "1"}]}


def test_braces_and_brackets_inside_string_values_do_not_end_the_scan():
    # The scanner must not treat a `}` inside a string literal as the end of the
    # object; a naive brace counter truncates this payload mid-string.
    text = 'Reply: {"msg": "a {b} [c] end", "n": 1} -- thanks'
    assert extract_json(text) == {"msg": "a {b} [c] end", "n": 1}


def test_escaped_quotes_inside_string_values_do_not_end_the_scan():
    # An escaped quote must not flip the scanner out of string mode.
    text = r'Reply: {"msg": "she said \"stop\" loudly"} done'
    assert extract_json(text) == {"msg": 'she said "stop" loudly'}


def test_escaped_backslash_before_quote_keeps_the_string_open():
    text = r'{"path": "C:\\", "tail": "}"}'
    assert extract_json(text) == {"path": "C:\\", "tail": "}"}


# -------------------------------------------------------------------- structures


def test_top_level_array_parses():
    assert extract_json('[{"id": "1"}, {"id": "2"}]') == [{"id": "1"}, {"id": "2"}]


def test_top_level_scalar_parses():
    assert extract_json("42") == 42


def test_nested_object_parses():
    text = 'Here you go: {"outer": {"inner": [1, {"deep": true}]}} bye'
    assert extract_json(text) == {"outer": {"inner": [1, {"deep": True}]}}


def test_first_of_several_json_objects_is_returned():
    # Only the first object is a payload; the rest is trailing commentary.
    text = '{"first": 1} then {"second": 2}'
    assert extract_json(text) == {"first": 1}


def test_first_of_several_fenced_blocks_is_returned():
    text = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
    assert extract_json(text) == {"first": 1}


# ------------------------------------------------------------------------ errors


def test_none_input_raises_value_error_naming_the_type():
    with pytest.raises(ValueError, match="NoneType"):
        extract_json(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [123, 4.5, True, ["a"], {"a": 1}])
def test_non_string_input_raises_value_error(value):
    with pytest.raises(ValueError, match="Expected str"):
        extract_json(value)  # type: ignore[arg-type]


def test_empty_string_raises_value_error():
    with pytest.raises(ValueError, match="No JSON found in empty response"):
        extract_json("")


def test_whitespace_only_string_raises_value_error():
    with pytest.raises(ValueError, match="No JSON found in empty response"):
        extract_json("   \n\t ")


def test_unparseable_text_raises_value_error_quoting_the_input():
    with pytest.raises(ValueError) as excinfo:
        extract_json("I am afraid I cannot help with that.")
    assert "I am afraid I cannot help with that." in str(excinfo.value)


def test_unbalanced_braces_raise_value_error_quoting_the_input():
    with pytest.raises(ValueError) as excinfo:
        extract_json("{oops, still prose")
    assert "{oops, still prose" in str(excinfo.value)


def test_mismatched_closing_bracket_is_not_treated_as_the_end_of_the_object():
    with pytest.raises(ValueError):
        extract_json('{"a": 1]')


def test_long_unparseable_input_is_truncated_in_the_error_message():
    text = "x" * 500
    with pytest.raises(ValueError) as excinfo:
        extract_json(text)
    assert "..." in str(excinfo.value)
