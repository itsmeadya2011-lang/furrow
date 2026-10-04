"""Tests for :mod:`furrow.parsing`.

The module's whole job is turning messy model text into a Python value, so these
tests feed it the shapes real models emit — bare JSON, markdown fences (with and
without a language tag, in either case), prose on either side, nested containers,
and braces inside string literals — and assert two invariants:

* a recoverable response always parses to the right value;
* an unrecoverable response always raises
  :class:`~furrow.errors.StructuredOutputError` (never a bare ``ValueError`` or
  an infinite scan), with the *full* raw response preserved on ``.raw`` even
  though ``str(err)`` echoes only a bounded prefix.

Everything is offline and deterministic: no network, no API key.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from furrow.config import Plan, TaskModel
from furrow.errors import FurrowError, StructuredOutputError
from furrow.parsing import (
    MAX_RAW_ECHO,
    extract_json,
    parse_json,
    parse_model,
    strip_code_fences,
)

# A JSON *string value* containing both braces and escaped quotes. Parsed, this
# must come back as a dict whose single value is the unescaped text -- the braces
# must not throw off the balanced-span scanner and the escapes must survive.
TRICKY_TEXT = '{"a": "he said {hi} \\"ok\\""}'
TRICKY_VALUE = {"a": 'he said {hi} "ok"'}


# --------------------------------------------------------------------------
# strip_code_fences
# --------------------------------------------------------------------------


def test_strip_code_fences_leaves_unfenced_text_unchanged() -> None:
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_strip_code_fences_extracts_a_json_tagged_block() -> None:
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_extracts_a_bare_fenced_block() -> None:
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_extracts_an_uppercase_json_tag() -> None:
    assert strip_code_fences('```JSON\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_ignores_prose_around_the_block() -> None:
    text = 'Sure, here you go:\n```json\n{"a": 1}\n```\nLet me know.'
    assert strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_handles_a_fence_closed_on_the_same_line() -> None:
    assert strip_code_fences('``` {"a": 1} ``` and then some prose') == '{"a": 1}'


# --------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------


def test_extract_json_returns_the_bare_payload() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_returns_the_fenced_payload() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_scans_out_a_prose_wrapped_payload() -> None:
    text = 'Here is the plan:\n{"a": 1}\nHope that helps!'
    assert json.loads(extract_json(text)) == {"a": 1}


# --------------------------------------------------------------------------
# parse_json: payloads real models actually emit
# --------------------------------------------------------------------------


def test_parse_json_bare_object() -> None:
    assert parse_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_json_json_fenced_object() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_bare_fenced_object() -> None:
    assert parse_json('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_uppercase_json_fenced_object() -> None:
    assert parse_json('```JSON\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_object_surrounded_by_prose_before_and_after() -> None:
    text = (
        "Certainly! Here is the plan you asked for:\n"
        '```json\n{"tasks": [], "rationale": "done"}\n```\n'
        "Let me know if you want anything changed."
    )
    assert parse_json(text) == {"tasks": [], "rationale": "done"}


def test_parse_json_prose_without_any_fence() -> None:
    text = 'Thinking out loud... the answer is {"a": [1, 2]} -- done.'
    assert parse_json(text) == {"a": [1, 2]}


def test_parse_json_nested_objects_and_arrays() -> None:
    payload = {
        "outer": {
            "inner": [1, 2, {"deep": [3, {"deeper": "value"}]}],
            "empty_object": {},
            "empty_array": [],
        }
    }
    assert parse_json(json.dumps(payload)) == payload


def test_parse_json_nested_payload_wrapped_in_prose() -> None:
    payload = {"plan": {"tasks": [{"id": "1", "files": ["a.py", "b.py"]}]}}
    assert parse_json(f'Result:\n```json\n{json.dumps(payload)}\n```') == payload


def test_parse_json_string_value_with_braces_and_escaped_quotes() -> None:
    assert parse_json(TRICKY_TEXT) == TRICKY_VALUE


def test_parse_json_string_value_with_braces_inside_prose() -> None:
    # Forces the balanced-span scanner (not the whole-text shortcut) to find the
    # object, so the braces *inside* the string literal must be skipped.
    assert parse_json(f'Sure thing: {TRICKY_TEXT} -- enjoy!') == TRICKY_VALUE


def test_parse_json_top_level_array() -> None:
    assert parse_json("[1, 2, 3]") == [1, 2, 3]


def test_parse_json_top_level_array_in_a_fence() -> None:
    assert parse_json('```json\n[{"id": "1"}, {"id": "2"}]\n```') == [
        {"id": "1"},
        {"id": "2"},
    ]


def test_parse_json_scalar_payloads() -> None:
    assert parse_json("42") == 42
    assert parse_json('"just a string"') == "just a string"


# --------------------------------------------------------------------------
# parse_json: failure modes
# --------------------------------------------------------------------------


def test_unterminated_object_raises() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_json('{"a": 1')
    assert excinfo.value.raw == '{"a": 1'


def test_unterminated_array_raises() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_json("[1, 2")
    assert excinfo.value.raw == "[1, 2"


def test_unterminated_fenced_object_raises() -> None:
    text = '```json\n{"a": 1'
    with pytest.raises(StructuredOutputError):
        parse_json(text)


def test_trailing_comma_raises() -> None:
    with pytest.raises(StructuredOutputError):
        parse_json('{"a": 1,}')


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        "",
        "   \n  ",
        "```json\nnot json either\n```",
        "Sorry, I cannot help with that request.",
    ],
    ids=["garbage", "empty", "whitespace", "fenced-garbage", "refusal"],
)
def test_garbage_input_raises_structured_output_error(text: str) -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_json(text)
    err = excinfo.value
    assert err.raw == text
    assert "Could not extract JSON" in str(err)
    assert isinstance(err, FurrowError)


# --------------------------------------------------------------------------
# The raw-response echo contract
# --------------------------------------------------------------------------


def test_max_raw_echo_is_the_documented_bound() -> None:
    assert MAX_RAW_ECHO == 500


def test_error_message_truncates_the_echo_but_raw_keeps_everything() -> None:
    raw = "x" * 5000
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_json(raw)
    err = excinfo.value

    # The full text is retained for programmatic/logging use...
    assert err.raw == raw
    assert len(err.raw) == 5000

    # ...while str(err) echoes at most ~500 chars of it.
    rendered = str(err)
    assert raw[:MAX_RAW_ECHO] in rendered
    assert raw[MAX_RAW_ECHO:] not in rendered
    assert len(rendered) < len(raw)
    # The bounded copy is also what lands in the rendered details mapping.
    assert err.details["raw"] == raw[:MAX_RAW_ECHO]
    assert len(err.details["raw"]) <= 500


def test_raw_is_truncated_for_a_garbage_model_response_too() -> None:
    raw = '{"tasks": [' + ", ".join(f'{{"id": "{i}"}}' for i in range(1000)) + "]"
    assert len(raw) > 5000
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_model(raw, Plan)
    assert excinfo.value.raw == raw
    assert excinfo.value.details["raw"] == raw[:MAX_RAW_ECHO]


# --------------------------------------------------------------------------
# parse_model
# --------------------------------------------------------------------------


def test_parse_model_success() -> None:
    plan = parse_model('{"tasks": [{"id": "1", "description": "d"}], "rationale": "r"}', Plan)
    assert isinstance(plan, Plan)
    assert [task.id for task in plan.tasks] == ["1"]
    assert plan.rationale == "r"


def test_parse_model_success_from_a_fenced_response_with_prose() -> None:
    text = (
        "Here is the plan:\n"
        '```json\n{"tasks": [{"id": "1", "description": "d"}], "rationale": "r"}\n```\n'
        "Anything else?"
    )
    plan = parse_model(text, Plan)
    assert plan.task_map()["1"].description == "d"
    assert not plan.is_empty


def test_parse_model_success_defaults_are_applied() -> None:
    plan = parse_model('{"tasks": [{"id": "1", "description": "d"}], "rationale": "r"}', Plan)
    assert plan.tasks[0].files == []
    assert plan.tasks[0].result is None


def test_parse_model_wrong_shape_raises() -> None:
    # A JSON array is a perfectly good JSON document but the wrong *shape* for Plan.
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_model("[1, 2, 3]", Plan)
    err = excinfo.value
    assert "did not match Plan" in str(err)
    assert err.raw == "[1, 2, 3]"
    assert isinstance(err.__cause__, ValidationError)
    assert isinstance(err, FurrowError)


def test_parse_model_scalar_payload_raises() -> None:
    with pytest.raises(StructuredOutputError):
        parse_model("42", Plan)


def test_parse_model_missing_required_field_raises() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_model('{"tasks": []}', Plan)  # no "rationale"
    err = excinfo.value
    assert "rationale" in str(err)
    assert err.raw == '{"tasks": []}'
    assert isinstance(err.__cause__, ValidationError)


def test_parse_model_wrong_field_type_raises() -> None:
    with pytest.raises(StructuredOutputError):
        parse_model('{"tasks": {"id": "1"}, "rationale": "r"}', Plan)


def test_parse_model_unparseable_text_raises_without_a_validation_cause() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_model("I refuse to answer in JSON.", Plan)
    # Nothing was extracted at all, so there is no pydantic failure to chain.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.raw == "I refuse to answer in JSON."


def test_parse_model_rejects_a_model_that_does_not_match_even_in_a_fence() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_model('```json\n{"unexpected": true}\n```', TaskModel)
    assert "did not match TaskModel" in str(excinfo.value)


class _Nested(BaseModel):
    """Local pydantic model used to prove ``parse_model`` is generic."""

    label: str
    payload: dict[str, Any]


def test_parse_model_validates_any_pydantic_model() -> None:
    parsed = parse_model(
        'fence: ```json\n{"label": "x", "payload": {"n": [1, 2]}}\n```',
        _Nested,
    )
    assert parsed.label == "x"
    assert parsed.payload == {"n": [1, 2]}


def test_parse_model_error_names_the_target_model() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_model('{"label": 1}', _Nested)
    assert "_Nested" in str(excinfo.value)
