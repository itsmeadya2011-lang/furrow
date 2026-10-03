"""Tests for :mod:`furrow.agents.parsing`.

LLM output is not machine-readable by default: it arrives wrapped in markdown
code fences, buried in prose, with braces inside string literals, and sometimes
cut off mid-object. ``parse_json_response`` exists so that callers get a parsed
value (or a useful ``ValueError``) instead of a constant ``json.JSONDecodeError``.

These tests drive the public ``parse_json_response`` entry point; the private
helpers are only exercised where they carry behavior of their own.
"""

from __future__ import annotations

import json

import pytest

from furrow.agents.parsing import parse_json_response


def test_plain_object_parses() -> None:
    assert parse_json_response('{"goal": "ship", "cycles": 2}') == {"goal": "ship", "cycles": 2}


def test_plain_array_parses() -> None:
    assert parse_json_response('["a", "b"]') == ["a", "b"]


def test_fenced_json_block_parses() -> None:
    text = '```json\n{"passed": true, "summary": "all good"}\n```'
    assert parse_json_response(text) == {"passed": True, "summary": "all good"}


def test_bare_fence_parses() -> None:
    text = '```\n{"passed": false}\n```'
    assert parse_json_response(text) == {"passed": False}


def test_uppercase_language_tag_parses() -> None:
    # The tag is just noise: everything after the opening marker is kept.
    text = '```JSON\n{"passed": true}\n```'
    assert parse_json_response(text) == {"passed": True}


def test_fence_followed_by_prose_parses() -> None:
    text = '```json\n{"cycles": 1}\n```\nLet me know if you want a different breakdown.'
    assert parse_json_response(text) == {"cycles": 1}


def test_json_embedded_in_prose_is_extracted() -> None:
    text = (
        'Sure! Here is the plan you asked for:\n'
        '{"tasks": [], "rationale": "small"}\n'
        'Hope that helps.'
    )
    assert parse_json_response(text) == {"tasks": [], "rationale": "small"}


def test_nested_objects_and_arrays() -> None:
    text = '{"a": {"b": [1, {"c": 2}]}, "d": [[], {"e": null}]}'
    assert parse_json_response(text) == {"a": {"b": [1, {"c": 2}]}, "d": [[], {"e": None}]}


def test_braces_inside_string_literal_do_not_unbalance_depth() -> None:
    # The depth counter must ignore brackets inside strings, otherwise this never
    # balances (or balances at the wrong place) and parsing fails.
    text = '{"summary": "use { and } carefully", "passed": true}'
    assert parse_json_response(text) == {"summary": "use { and } carefully", "passed": True}


def test_brackets_inside_string_literal_do_not_unbalance_depth() -> None:
    text = '{"note": "list looks like [this] and (that)", "ok": true}'
    assert parse_json_response(text) == {"note": "list looks like [this] and (that)", "ok": True}


def test_escaped_quote_inside_string_parses() -> None:
    # The \" must not be treated as the end of the string, otherwise the {ok}
    # braces that follow would be counted as real structure.
    text = '{"s": "he said \\"hi\\" {ok}"}'
    assert parse_json_response(text) == {"s": 'he said "hi" {ok}'}


def test_escaped_backslash_before_quote_does_not_close_string() -> None:
    # A literal backslash terminates the escape: "a\\\\" ends the string here.
    text = '{"path": "C:\\\\", "rest": 1}'
    assert parse_json_response(text) == {"path": "C:\\", "rest": 1}


def test_first_of_multiple_objects_wins() -> None:
    text = '{"first": 1}\nAnd here is a second one: {"second": 2}'
    assert parse_json_response(text) == {"first": 1}


def test_first_object_wins_with_trailing_text() -> None:
    assert parse_json_response('{"a": 1}\ntrailing {"b": 2}') == {"a": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('Use {curly} braces like this: {"a": 1}', {"a": 1}),
        ("Prose with {stray} {balanced} {groups}: [1, 2]", [1, 2]),
        ('```json\n{placeholder} then\n{"passed": true}\n```', {"passed": True}),
    ],
)
def test_stray_invalid_braces_in_prose_do_not_hide_later_json(
    text: str, expected: object
) -> None:
    # Regression: a balanced-but-invalid candidate used to abort the entire
    # scan, so perfectly valid JSON sitting right after it was never reached.
    assert parse_json_response(text) == expected


def test_invalid_json_with_no_valid_json_after_it_still_raises() -> None:
    # Resuming after a failed candidate must terminate, and the error snippet
    # must still come from the original text, not the cleaned one.
    text = "Use {curly} braces like this, but here: {not json}"
    with pytest.raises(ValueError) as excinfo:
        parse_json_response(text)
    message = str(excinfo.value)
    assert "No valid JSON" in message
    assert repr(text) in message


def test_truncated_nested_object_raises_instead_of_returning_inner_slice() -> None:
    # The outer braces never balance, so there is no complete candidate at all:
    # the inner {"b": 1} must not be mistaken for the whole JSON response.
    with pytest.raises(ValueError):
        parse_json_response('{"a": {"b": 1}')


def test_json_with_newlines_and_unicode() -> None:
    text = '{\n  "summary": "café — ok",\n  "failures": ["a\\"b"]\n}'
    assert parse_json_response(text) == json.loads(text)


def test_text_without_json_raises_with_snippet() -> None:
    text = "I could not complete the task, sorry."
    with pytest.raises(ValueError) as excinfo:
        parse_json_response(text)
    message = str(excinfo.value)
    assert "No valid JSON" in message
    assert repr(text) in message


def test_error_snippet_comes_from_original_text_not_cleaned_text() -> None:
    # Fence markers are stripped before scanning, but the debug snippet must show
    # what the model actually returned, markers included.
    text = "```json\nnot json at all\n```"
    with pytest.raises(ValueError) as excinfo:
        parse_json_response(text)
    assert repr(text) in str(excinfo.value)


def test_incomplete_json_raises() -> None:
    # Response cut off mid-object: no balanced slice exists.
    with pytest.raises(ValueError):
        parse_json_response('{"summary": "half a respo')


def test_empty_string_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_response("")


@pytest.mark.parametrize("bad", [123, None, 1.5, b'{"a": 1}', ["x"], {"a": 1}])
def test_non_string_input_raises_value_error(bad: object) -> None:
    with pytest.raises(ValueError, match="Expected a string response"):
        parse_json_response(bad)  # type: ignore[arg-type]


def test_long_response_snippet_is_truncated() -> None:
    text = "x" * 900
    with pytest.raises(ValueError) as excinfo:
        parse_json_response(text)
    message = str(excinfo.value)
    assert "(truncated)" in message
    # The snippet is cut well below the full text so the message stays debuggable.
    assert message.count("x") == 500
    assert text not in message


def test_short_response_snippet_is_not_truncated() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_json_response("short and useless")
    assert "(truncated)" not in str(excinfo.value)


def test_invalid_json_inside_fence_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_response('```json\n{"trailing": }\n```')
