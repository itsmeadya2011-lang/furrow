"""Tests for :func:`furrow.agents.base.extract_json`.

``extract_json`` is the seam between "the model returned JSON" and "the model
returned markdown that happens to contain JSON", so every case below is a shape
an LLM actually produces: fenced blocks, chatty prose on both sides, braces inside
string values, and truncated output from a cut-off response.

The contract asserted throughout is the one the agents rely on: a usable payload is
returned on its own, whether it is a top-level object or a top-level array, and
anything the scanner cannot find a balanced ``{...}``/``[...]`` span in comes back
essentially unchanged so the caller can report it as unparseable.
"""

from __future__ import annotations

import json

import pytest

from furrow.agents.base import extract_json

_PLAIN = '{"a": 1}'


def test_plain_object_is_returned_unchanged() -> None:
    assert extract_json(_PLAIN) == _PLAIN


@pytest.mark.parametrize(
    "payload",
    [
        _PLAIN,
        '{"a": 1, "b": "two", "c": null}',
        '{"tasks": [], "rationale": "ship it"}',
    ],
)
def test_objects_without_wrappers_pass_through(payload: str) -> None:
    assert extract_json(payload) == payload


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        '```JSON\n{"a": 1}\n```',
        '```json\n  {"a": 1}  \n\n```',
    ],
    ids=["json-tag", "bare-tag", "uppercase-tag", "padded-body"],
)
def test_fenced_payload_is_unwrapped(response: str) -> None:
    assert extract_json(response) == _PLAIN


@pytest.mark.parametrize(
    "response",
    [
        'Here is the plan:\n{"a": 1}\nLet me know what you think.',
        'Sure!\n\n```json\n{"a": 1}\n```\n\nHope that helps.',
        'Thinking about it {"a": 1} in the end.',
    ],
    ids=["prose-around", "fenced-with-prose", "inline-object"],
)
def test_prose_around_the_object_is_discarded(response: str) -> None:
    assert extract_json(response) == _PLAIN


def test_nested_objects_and_arrays_survive_extraction() -> None:
    payload = '{"outer": {"inner": [1, 2, {"deep": true}]}, "sibling": {"x": []}}'
    extracted = extract_json(f'Result:\n{payload}\ndone')
    assert extracted == payload
    assert json.loads(extracted) == json.loads(payload)


def test_brace_inside_a_string_value_does_not_break_nesting() -> None:
    payload = '{"a": "{not real}", "b": 2}'
    extracted = extract_json(f'Template: {payload} end')
    assert extracted == payload
    # The naive "count every brace" reader would stop at the brace inside the string
    # and hand back a fragment that does not parse.
    assert json.loads(extracted) == {"a": "{not real}", "b": 2}


def test_escaped_quote_inside_a_string_does_not_end_it() -> None:
    payload = '{"a": "he said \\"hi\\"", "b": 2}'
    extracted = extract_json(f'Result:\n{payload}')
    assert extracted == payload
    assert json.loads(extracted)["a"] == 'he said "hi"'


def test_escaped_backslash_before_a_closing_quote_does_not_end_the_string() -> None:
    payload = '{"a": "c:\\\\", "b": 2}'
    extracted = extract_json(payload)
    assert extracted == payload
    # The string value ends in a single backslash, so the quote after it really is
    # the closing quote; treating the backslash as escaping that quote would send
    # the scanner looking for a brace that is not there.
    assert json.loads(extracted)["a"] == "c:\\"


def test_first_object_span_wins_over_later_ones() -> None:
    assert extract_json('```json\n{"a": 1}\n{"b": 2}\n```') == _PLAIN


def test_top_level_array_of_objects_is_returned_whole() -> None:
    # An array of tasks is a plausible planner reply. Scanning for the first "{"
    # span would hand back only the first element and every later task would vanish.
    payload = '[{"id": "1", "description": "one"}, {"id": "2", "description": "two"}]'
    extracted = extract_json(f'Here are the tasks:\n{payload}\nLet me know.')
    assert extracted == payload
    assert json.loads(extracted) == [
        {"id": "1", "description": "one"},
        {"id": "2", "description": "two"},
    ]


def test_fenced_top_level_array_is_unwrapped_whole() -> None:
    payload = '[{"id": "1"}, {"id": "2"}]'
    extracted = extract_json(f'```json\n{payload}\n```')
    assert extracted == payload
    assert len(json.loads(extracted)) == 2


def test_nested_array_inside_an_object_still_extracts_whole() -> None:
    payload = '{"tasks": [{"id": "1"}, {"id": "2"}], "rationale": "ship it"}'
    extracted = extract_json(f'```json\n{payload}\n```')
    assert extracted == payload
    assert json.loads(extracted)["tasks"] == [{"id": "1"}, {"id": "2"}]


def test_brace_inside_a_string_in_an_array_element_does_not_break_the_scan() -> None:
    # Both a stray "}" and a stray "{" live inside string values here, so a reader
    # that counted brackets without tracking string state would stop in the wrong
    # place and return a fragment that does not parse.
    payload = '[{"a": "}{"}, {"b": "{{"}]'
    extracted = extract_json(f'Result:\n{payload}\nend')
    assert extracted == payload
    assert json.loads(extracted) == [{"a": "}{"}, {"b": "{{"}]


def test_truncated_top_level_array_is_returned_whole_so_it_fails_closed() -> None:
    # The array never closes, so there is no balanced span to hand back. Returning
    # the whole (unparseable) response is the point: the old scanner returned the
    # first element, which parsed but silently dropped the rest of the plan.
    response = '[{"id": "1"}'
    assert extract_json(response) == response
    with pytest.raises(json.JSONDecodeError, match="Expecting"):
        json.loads(response)


def test_first_brace_wins_even_when_it_is_prose() -> None:
    # Documented limitation rather than a promise: the scanner anchors on the first
    # "{" in the response, so a brace in the surrounding chat is picked up.
    assert extract_json("Use {curly} braces: " + _PLAIN) == "{curly}"


def test_empty_string_is_returned_unchanged() -> None:
    result = extract_json("")
    assert result == ""
    assert not result


def test_whitespace_only_input_yields_an_empty_string() -> None:
    assert extract_json("   \n\t ") == ""


def test_lone_fence_marker_has_no_body_to_unwrap() -> None:
    assert extract_json("```") == ""


@pytest.mark.parametrize(
    "response",
    [
        "no json here at all",
        "closing } brace only",
        "[1, 2, 3]",
        "",
    ],
    ids=["plain-prose", "lone-closing-brace", "array", "empty"],
)
def test_text_without_an_object_span_is_returned_unchanged(response: str) -> None:
    assert extract_json(response) == response


def test_fenced_text_without_an_object_span_comes_back_fence_stripped() -> None:
    assert extract_json('```json\nno object in this block\n```') == "no object in this block"


@pytest.mark.parametrize(
    "response",
    [
        '{"a": 1',
        '{"a": {"b": 1',
        '{"a": "unterminated',
    ],
    ids=["unclosed-object", "unclosed-nested", "unterminated-string"],
)
def test_truncated_json_is_returned_unchanged(response: str) -> None:
    assert extract_json(response) == response


def test_truncated_json_in_a_fence_comes_back_as_the_fence_stripped_fragment() -> None:
    # The agents use this: whatever cannot be parsed is surfaced in the error
    # message instead of being swallowed.
    assert extract_json('```json\n{"a": 1\n```') == '{"a": 1'
