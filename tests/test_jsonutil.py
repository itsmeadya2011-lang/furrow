"""Contract tests for :mod:`furrow.jsonutil`, the LLM-output parsing layer.

Every model response in furrow goes through here: an LLM hands back markdown-fenced JSON,
or JSON behind a sentence of preamble, or JSON with a chatty sign-off, and all three must
produce the same object. Two properties matter beyond "it parses":

* **Failures are bounded.** A model that returns 8 000 characters of prose must not end up
  quoted verbatim inside a ``ValueError``; :func:`truncate` caps it.
* **Failures are chained.** ``parse_model_json`` re-raises as a single ``ValueError`` naming
  the target model and the offending fields, with the original
  :class:`pydantic.ValidationError` on ``__cause__`` so the traceback keeps the detail.

The first block is the behaviour that already existed and was proven in isolation; the rest
covers the fence/prose, nesting and error-reporting edges. Ported from the scratch checks in
``/tmp/kilo/test_jsonutil_scratch.py``, which must keep passing unchanged.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from furrow import jsonutil
from furrow.jsonutil import extract_json, parse_model_json, strip_code_fences, truncate
from furrow.models import Plan, TaskModel

# --------------------------------------------------------------------------- strip_code_fences


def test_strip_fenced_json_tag():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_bare_fence_and_crlf():
    assert strip_code_fences('```\r\n{"a": 1}\r\n```\r\n') == '{"a": 1}'


def test_strip_unterminated_and_unfenced():
    assert strip_code_fences('```json\n{"a": 1}') == '{"a": 1}'
    assert strip_code_fences("no fence here") == "no fence here"
    assert strip_code_fences("") == ""


def test_strip_accepts_a_fenced_language_tag_with_trailing_space():
    assert strip_code_fences('```JSON  \n{"a": 1}\n```') == '{"a": 1}'


def test_strip_leaves_a_mid_prose_backtick_run_alone():
    # The opening fence is anchored at the start and the closing fence at the end, so a
    # ``` that appears inside the payload must survive.
    assert strip_code_fences('text with ``` inside') == "text with ``` inside"


# ------------------------------------------------------------------------------- extract_json


def test_extract_fenced_and_tagged():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n[1, 2]\n```') == [1, 2]


def test_extract_leading_prose_before_fence_and_bare():
    fenced = "Here is the plan:\n```json\n{\"a\": 1}\n```\nHope that helps!"
    assert extract_json(fenced) == {"a": 1}
    assert extract_json('Here is the plan: {"a": 1} Hope that helps!') == {"a": 1}


def test_extract_tolerates_prose_and_nested_braces():
    text = 'Preface with a stray { brace. Then: {"a": {"b": 2}} and trailing prose.'
    assert extract_json(text) == {"a": {"b": 2}}


def test_extract_top_level_array_and_invalid_inputs():
    assert extract_json('```json\n[{"id": "a"}]\n```') == [{"id": "a"}]
    for bad in ("", "   ", "just some prose, no json at all"):
        with pytest.raises(ValueError):
            extract_json(bad)
    with pytest.raises(ValueError):
        extract_json("```json\n{'a': 1}\n```")  # single quotes are not JSON


def test_extract_json_error_snippet_is_truncated():
    with pytest.raises(ValueError) as excinfo:
        extract_json("x" * 5000)
    message = str(excinfo.value)
    assert len(message) < 600
    assert "chars truncated" in message


def test_extract_handles_crlf_line_endings_inside_a_fence():
    text = '```json\r\n{"a": 1, "b": [1, 2]}\r\n```\r\n'
    assert strip_code_fences(text) == '{"a": 1, "b": [1, 2]}'
    assert extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_handles_crlf_fence_with_a_bare_tag():
    assert extract_json('```\r\n{"a": 1}\r\n```') == {"a": 1}


def test_extract_finds_the_object_past_a_brace_in_the_leading_prose():
    """A `{` in the preamble must not win: only a *decodable* value is accepted.

    The parser walks every `{`/`[` in turn and keeps the first index that actually decodes,
    so an unbalanced or non-JSON brace in the prose is skipped instead of ending the search.
    """
    assert extract_json('I will merge { this } into one file. Here: {"a": 1}') == {"a": 1}
    assert extract_json('Use a dict { like this.\n{"a": 1}\nEnjoy!') == {"a": 1}
    assert extract_json('An array [ like this ] then {"a": 1}') == {"a": 1}
    # The prose brace must not be mistaken for the payload either way round.
    assert extract_json('{"a": 1} is the shape, roughly { like this }') == {"a": 1}


def test_extract_preserves_nesting_and_scalar_types():
    payload = {
        "plan": {"tasks": [{"id": "1", "steps": [{"n": 1}, {"n": 2}]}]},
        "ok": True,
        "none": None,
        "delta": -1.5,
    }
    text = f"```json\n{json.dumps(payload)}\n```"

    assert extract_json(text) == payload


# --------------------------------------------------------------------------------- parse_model_json


def test_parse_model_json_success_and_validation_failure():
    payload = json.dumps({"tasks": [{"id": "a", "description": "do it"}], "rationale": "because"})
    plan = parse_model_json(f"```json\n{payload}\n```", Plan)
    assert isinstance(plan, Plan)
    assert plan.tasks[0].id == "a"
    assert plan.tasks[0].files == []

    with pytest.raises(ValueError) as excinfo:
        parse_model_json('{"tasks": [{"id": "a"}]}', Plan)
    assert "Plan" in str(excinfo.value)
    assert "description" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None

    with pytest.raises(ValueError) as excinfo:
        parse_model_json("not json at all", TaskModel)
    assert "TaskModel" in str(excinfo.value)


def test_parse_model_json_default_rationale_tolerated():
    plan = parse_model_json('{"tasks": []}', Plan)
    assert plan.rationale == ""


def test_parse_model_json_error_names_the_model_and_the_offending_field():
    with pytest.raises(ValueError) as excinfo:
        parse_model_json('{"tasks": [{"id": "a"}]}', Plan)
    message = str(excinfo.value)

    assert message.startswith("model output did not match Plan:")
    assert "tasks.0.description: Field required" in message


def test_parse_model_json_chains_the_original_validation_error():
    with pytest.raises(ValueError) as excinfo:
        parse_model_json('{"tasks": [{"id": "a"}]}', Plan)

    cause = excinfo.value.__cause__
    assert isinstance(cause, ValidationError)


def test_parse_model_json_chains_the_original_decode_error():
    with pytest.raises(ValueError) as excinfo:
        parse_model_json("I would love to help with that!", TaskModel)
    message = str(excinfo.value)

    assert message.startswith("could not parse TaskModel from model output:")
    # the underlying complaint is preserved rather than replaced
    assert "no JSON value found in model output" in message
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_parse_model_json_preserves_valid_input():
    task = parse_model_json('```json\n{"id": "1", "description": "d"}\n```', TaskModel)

    assert isinstance(task, TaskModel)
    assert task.id == "1"
    assert task.status.value == "pending"


# ---------------------------------------------------------------------------------------- truncate


def test_truncate():
    assert truncate("hello", 10) == "hello"
    assert truncate("hello", 5) == "hello"
    assert truncate("hello", 2) == "he... (3 chars truncated)"


def test_truncate_is_a_no_op_at_or_under_the_limit():
    assert truncate("", 0) == ""
    assert truncate("abc", 3) == "abc"
    assert truncate("abc", 4) == "abc"
    assert truncate("x" * 2000) == "x" * 2000  # the default limit


def test_truncate_appends_a_marker_that_counts_the_dropped_characters():
    assert truncate("abcdefghij", 4) == "abcd... (6 chars truncated)"
    # the marker is additive, so the result is longer than the limit
    assert len(truncate("abcdefghij", 4)) > 4
    assert truncate("x" * 2001).endswith("... (1 chars truncated)")


def test_jsonutil_exports_exactly_the_documented_surface():
    assert jsonutil.__all__ == ["extract_json", "parse_model_json", "strip_code_fences", "truncate"]
    for name in jsonutil.__all__:
        assert getattr(jsonutil, name) is not None
