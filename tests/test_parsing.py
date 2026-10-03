"""Tests for :mod:`furrow.agents.parsing`, the tolerant LLM-output reader.

This layer exists because bare ``json.loads`` was killing cycles: real models
wrap payloads in ```` ```json ```` fences, prefix them with "Here is the plan:",
suffix them with commentary, or encode them two or three times over. The hard
case is the hand-written depth scanner -- a JSON string value may legally contain
``}``, ``{``, ``[`` and escaped quotes, and a naive ``text[text.index("{"):]``
would truncate it.

The other recurring real-world case: ``"3 passed, 1 error"`` is a failure, not a
pass, while ``"5 passed, no failures"`` is not a failure at all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from furrow.agents.parsing import (
    MAX_EXCERPT_CHARS,
    coerce_plan,
    coerce_test_result,
    excerpt,
    extract_json_object,
    looks_like_failure,
    looks_like_pass,
    parse_json_response,
    strip_code_fences,
)
from furrow.config import TaskStatus

# A string value full of characters that would break a naive brace scan.
TRICKY_PAYLOAD: dict[str, Any] = {
    "rationale": 'Rewrite `if (x) { y([1, 2]) }` and handle the "quoted" case',
    "tasks": [
        {
            "id": "t1",
            "description": 'fix path C:\\tmp\\"weird"\\name',
            "files": ["a{b}c.py"],
        }
    ],
}
TRICKY_JSON = json.dumps(TRICKY_PAYLOAD)


# --------------------------------------------------------------------------- #
# strip_code_fences
# --------------------------------------------------------------------------- #


def test_strip_code_fences_returns_the_body_of_a_json_fence() -> None:
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_returns_the_body_of_a_bare_fence() -> None:
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_keeps_an_unterminated_fence_body() -> None:
    assert strip_code_fences('```json\n{"a": 1}') == '{"a": 1}'


def test_strip_code_fences_ignores_prose_around_the_fence() -> None:
    text = 'Here you go:\n```json\n{"a": 1}\n```\nLet me know!'
    assert strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_strips_text_that_has_no_fence() -> None:
    assert strip_code_fences('  {"a": 1}  ') == '{"a": 1}'


# --------------------------------------------------------------------------- #
# extract_json_object
# --------------------------------------------------------------------------- #


def test_extract_json_object_reads_clean_json() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_reads_fenced_json() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_object_skips_a_prose_prefix() -> None:
    assert extract_json_object('Here is the plan: {"a": 1}') == {"a": 1}


def test_extract_json_object_skips_a_prose_suffix() -> None:
    assert extract_json_object('{"a": 1}\nHope that helps!') == {"a": 1}


def test_extract_json_object_skips_prose_on_both_sides() -> None:
    text = 'Sure!\n{"a": 1}\nAnything else?'
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_recovers_braces_inside_a_string_value() -> None:
    assert extract_json_object(f'Prose before: {TRICKY_JSON} and after.') == TRICKY_PAYLOAD


def test_extract_json_object_reads_a_top_level_array() -> None:
    assert extract_json_object('[{"a": 1}, 2]') == [{"a": 1}, 2]


def test_extract_json_object_reads_a_fenced_top_level_array() -> None:
    assert extract_json_object('```json\n[1, 2, 3]\n```') == [1, 2, 3]


def test_extract_json_object_reads_a_nested_array_from_prose() -> None:
    assert extract_json_object('Results: [1, [2, 3]] done') == [1, [2, 3]]


def test_extract_json_object_skips_an_unbalanced_leading_brace() -> None:
    assert extract_json_object('{oops then {"a": 1}') == {"a": 1}


def test_extract_json_object_rejects_text_with_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object or array found"):
        extract_json_object("Sorry, I cannot help with that.")


def test_extract_json_object_error_carries_an_excerpt_of_the_response() -> None:
    with pytest.raises(ValueError, match="I am not going to do that"):
        extract_json_object("I am not going to do that.")


def test_extract_json_object_error_truncates_a_long_response() -> None:
    with pytest.raises(ValueError, match=r"\.\.\.\[truncated\]"):
        extract_json_object("no json here " * 200)


# --------------------------------------------------------------------------- #
# parse_json_response
# --------------------------------------------------------------------------- #


def test_parse_json_response_reads_clean_json() -> None:
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_reads_fenced_json_with_prose() -> None:
    text = f'The plan:\n```json\n{TRICKY_JSON}\n```\nDone.'
    assert parse_json_response(text) == TRICKY_PAYLOAD


def test_parse_json_response_unwraps_double_encoded_json() -> None:
    once = json.dumps({"a": 1})
    assert parse_json_response(json.dumps(once)) == {"a": 1}


def test_parse_json_response_unwraps_triple_encoded_json() -> None:
    once = json.dumps({"a": 1})
    twice = json.dumps(once)
    assert parse_json_response(json.dumps(twice)) == {"a": 1}


def test_parse_json_response_unwraps_a_json_string_wrapped_in_a_fence() -> None:
    once = json.dumps({"a": 1})
    assert parse_json_response(f"```json\n{once}\n```") == {"a": 1}


def test_parse_json_response_rejects_plain_prose() -> None:
    with pytest.raises(ValueError, match="no JSON object or array found"):
        parse_json_response("There is nothing structured here.")


# --------------------------------------------------------------------------- #
# coerce_plan
# --------------------------------------------------------------------------- #


def test_coerce_plan_reads_a_well_formed_plan() -> None:
    plan = coerce_plan(
        {
            "rationale": "two small steps",
            "tasks": [
                {"id": "t1", "description": "first", "files": ["a.py"]},
                {"id": "t2", "description": "second", "dependencies": ["t1"]},
            ],
        }
    )
    assert plan.rationale == "two small steps"
    assert [task.id for task in plan.tasks] == ["t1", "t2"]
    assert plan.tasks[0].files == ["a.py"]
    assert plan.tasks[1].dependencies == ["t1"]
    assert plan.tasks[0].status == TaskStatus.PENDING


def test_coerce_plan_defaults_a_missing_rationale() -> None:
    plan = coerce_plan({"tasks": [{"id": "t1", "description": "first"}]})
    assert plan.rationale == ""


def test_coerce_plan_ignores_unknown_keys() -> None:
    payload = {
        "rationale": "r",
        "confidence": 0.9,
        "tasks": [{"id": "t1", "description": "d", "eta": 5}],
    }
    plan = coerce_plan(payload)
    assert plan.rationale == "r"
    assert len(plan.tasks) == 1


def test_coerce_plan_turns_a_bare_string_files_value_into_a_list() -> None:
    plan = coerce_plan({"tasks": [{"id": "t1", "description": "d", "files": "a.py"}]})
    assert plan.tasks[0].files == ["a.py"]


def test_coerce_plan_coerces_a_numeric_task_id() -> None:
    plan = coerce_plan({"tasks": [{"id": 7, "description": "d"}]})
    assert plan.tasks[0].id == "7"


def test_coerce_plan_substitutes_a_positional_id_when_the_model_omits_one() -> None:
    plan = coerce_plan({"tasks": [{"description": "first"}, {"description": "second"}]})
    assert [task.id for task in plan.tasks] == ["1", "2"]


def test_coerce_plan_accepts_a_bare_list_of_tasks() -> None:
    plan = coerce_plan([{"id": "t1", "description": "first"}])
    assert [task.id for task in plan.tasks] == ["t1"]
    assert plan.rationale == ""


def test_coerce_plan_accepts_a_single_task_mapping() -> None:
    plan = coerce_plan({"id": "t1", "description": "lonely task"})
    assert [task.id for task in plan.tasks] == ["t1"]


def test_coerce_plan_parses_a_payload_that_arrived_as_a_json_string() -> None:
    plan = coerce_plan(json.dumps({"tasks": [{"id": "t1", "description": "first"}]}))
    assert [task.id for task in plan.tasks] == ["t1"]


def test_coerce_plan_defaults_an_unrecognised_status_to_pending() -> None:
    plan = coerce_plan({"tasks": [{"id": "t1", "description": "d", "status": "done-ish"}]})
    assert plan.tasks[0].status == TaskStatus.PENDING


def test_coerce_plan_keeps_a_recognised_status() -> None:
    plan = coerce_plan({"tasks": [{"id": "t1", "description": "d", "status": "completed"}]})
    assert plan.tasks[0].status == TaskStatus.COMPLETED


def test_coerce_plan_keeps_a_bare_string_description() -> None:
    plan = coerce_plan(["just do the thing"])
    assert [task.description for task in plan.tasks] == ["just do the thing"]


def test_coerce_plan_rejects_a_scalar_payload() -> None:
    with pytest.raises(ValueError, match="plan payload must be an object or a list"):
        coerce_plan(42)


def test_coerce_plan_rejects_a_string_without_json() -> None:
    with pytest.raises(ValueError, match="does not contain JSON"):
        coerce_plan("no json in here")


# --------------------------------------------------------------------------- #
# coerce_test_result
# --------------------------------------------------------------------------- #


def test_coerce_test_result_reads_a_well_formed_verdict() -> None:
    result = coerce_test_result({"passed": True, "summary": "all green", "failures": []})
    assert result.passed is True
    assert result.summary == "all green"
    assert result.failures == []


def test_coerce_test_result_defaults_a_missing_summary() -> None:
    result = coerce_test_result({"passed": True})
    assert result.passed is True
    assert result.summary == ""


def test_coerce_test_result_defaults_missing_failures() -> None:
    result = coerce_test_result({"passed": False, "summary": "boom"})
    assert result.failures == []


def test_coerce_test_result_coerces_a_stringly_typed_passed_flag() -> None:
    assert coerce_test_result({"passed": "yes"}).passed is True
    assert coerce_test_result({"passed": "no"}).passed is False


def test_coerce_test_result_coerces_a_numeric_passed_flag() -> None:
    assert coerce_test_result({"passed": 1}).passed is True
    assert coerce_test_result({"passed": 0}).passed is False


def test_coerce_test_result_accepts_errors_as_a_synonym_for_failures() -> None:
    result = coerce_test_result({"passed": False, "errors": ["tests/test_a.py"]})
    assert result.failures == ["tests/test_a.py"]


def test_coerce_test_result_derives_the_verdict_from_prose() -> None:
    result = coerce_test_result("3 passed, 1 error")
    assert result.passed is False
    assert result.summary == "3 passed, 1 error"


def test_coerce_test_result_treats_an_empty_payload_as_a_failure() -> None:
    result = coerce_test_result({})
    assert result.passed is False


def test_coerce_test_result_rejects_a_scalar_payload() -> None:
    with pytest.raises(ValueError, match="test result payload must be an object"):
        coerce_test_result(42)


# --------------------------------------------------------------------------- #
# looks_like_failure / looks_like_pass
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "3 passed, 1 error",
        "FAILED tests/test_a.py::test_b - boom",
        "AssertionError: expected 1 to equal 2",
        "Traceback (most recent call last):",
        "2 tests failed",
    ],
)
def test_looks_like_failure_detects_failure_wording(text: str) -> None:
    assert looks_like_failure(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "5 passed, no failures",
        "0 failed, 12 passed",
        "0 errors, 4 tests",
        "All tests passed",
    ],
)
def test_looks_like_failure_ignores_negated_or_clean_output(text: str) -> None:
    assert looks_like_failure(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "All tests passed",
        "12 passed in 0.4s",
        "Everything is green",
        "Build success",
    ],
)
def test_looks_like_pass_detects_success_wording(text: str) -> None:
    assert looks_like_pass(text) is True


@pytest.mark.parametrize(
    "text",
    ["No test runner found.", "", "I could not determine the outcome"],
)
def test_looks_like_pass_is_false_without_an_explicit_signal(text: str) -> None:
    assert looks_like_pass(text) is False


def test_looks_like_pass_and_failure_can_both_be_true() -> None:
    text = "2 failed, 3 passed"
    assert looks_like_failure(text) is True
    assert looks_like_pass(text) is True


# --------------------------------------------------------------------------- #
# excerpt
# --------------------------------------------------------------------------- #


def test_excerpt_flattens_whitespace() -> None:
    assert excerpt("a\n\n  b\tc ") == "a b c"


def test_excerpt_leaves_short_text_untouched() -> None:
    assert excerpt("short") == "short"


def test_excerpt_truncates_at_the_default_limit() -> None:
    assert excerpt("x" * (MAX_EXCERPT_CHARS + 50)).endswith("...[truncated]")
    assert len(excerpt("x" * 10_000)) == MAX_EXCERPT_CHARS + len("...[truncated]")


def test_excerpt_honours_an_explicit_limit() -> None:
    assert excerpt("abcdefghij", limit=3) == "abc...[truncated]"
