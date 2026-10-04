"""Recovering JSON from LLM output.

Providers wrap JSON in markdown fences, bury it in prose and put braces inside string
values. A naive regex fails on all three and the failure used to kill a whole cycle, so
each wrapping style is pinned here, together with the requirement that an error message
carries a snippet of the offending text.
"""

from __future__ import annotations

import json

import pytest

from furrow.agents.jsonutil import extract_json, parse_json_model
from furrow.config import Plan, TaskModel
from furrow.config import TestResult as Verdict

BARE = '{"passed": true, "summary": "1 passed"}'
FENCED_JSON = f"```json\n{BARE}\n```"
FENCED_BARE = f"```\n{BARE}\n```"
IN_PROSE = f"Here is the verdict you asked for:\n{BARE}\nLet me know if that helps."
WITH_BRACES_IN_STRINGS = (
    '{"summary": "a } b { c", "passed": false, "failures": ["x} y", {"detail": "nested } brace"}]}'
)
ARRAY = '[{"id": "1"}, {"id": "2"}]'


def test_bare_json_is_returned_unchanged() -> None:
    assert extract_json(BARE) == BARE


def test_json_fence_is_stripped() -> None:
    assert json.loads(extract_json(FENCED_JSON))["summary"] == "1 passed"


def test_bare_code_fence_is_stripped() -> None:
    assert json.loads(extract_json(FENCED_BARE))["summary"] == "1 passed"


def test_json_wrapped_in_prose_is_found() -> None:
    assert json.loads(extract_json(IN_PROSE))["summary"] == "1 passed"


def test_trailing_prose_after_the_json_is_ignored() -> None:
    assert extract_json(f"{BARE}\n\nAnything else you want to know?") == BARE


def test_braces_inside_string_values_do_not_end_the_extraction() -> None:
    extracted = extract_json(f"prose {WITH_BRACES_IN_STRINGS} more prose")
    assert json.loads(extracted)["summary"] == "a } b { c"


def test_nested_objects_and_arrays_survive_extraction() -> None:
    data = json.loads(extract_json(WITH_BRACES_IN_STRINGS))
    assert data["failures"][1] == {"detail": "nested } brace"}


def test_top_level_array_is_extracted() -> None:
    assert [item["id"] for item in json.loads(extract_json(ARRAY))] == ["1", "2"]


def test_escaped_quote_does_not_end_a_string() -> None:
    text = '{"summary": "he said \\"stop\\" then { more", "passed": false}'
    assert json.loads(extract_json(text))["passed"] is False


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty or non-string"):
        extract_json("   ")


def test_text_without_any_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="No JSON value found"):
        extract_json("I am afraid I cannot do that")


def test_fence_without_content_is_rejected() -> None:
    with pytest.raises(ValueError, match="No JSON value found"):
        extract_json("```json\n```")


def test_unbalanced_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unbalanced JSON"):
        extract_json('{"passed": true')


def test_error_message_quotes_the_offending_text() -> None:
    with pytest.raises(ValueError) as excinfo:
        extract_json("I am afraid I cannot do that")
    assert "I am afraid I cannot do that" in str(excinfo.value)


def test_error_message_truncates_a_very_long_offender() -> None:
    with pytest.raises(ValueError) as excinfo:
        extract_json("x" * 4000)
    message = str(excinfo.value)
    assert message.endswith("…'")
    assert len(message) < 600
    assert "x" * 1000 not in message


def test_error_message_keeps_a_short_offender_whole() -> None:
    with pytest.raises(ValueError) as excinfo:
        extract_json("nope")
    assert "nope" in str(excinfo.value)


def test_parse_json_model_builds_the_model_from_fenced_json() -> None:
    result = parse_json_model(FENCED_JSON, Verdict)
    assert isinstance(result, Verdict)
    assert (result.passed, result.summary) == (True, "1 passed")


def test_parse_json_model_builds_a_plan_from_prose_wrapped_json() -> None:
    payload = json.dumps(
        {
            "tasks": [{"id": "1", "description": "do it", "files": ["a.py"]}],
            "rationale": "one step",
        }
    )
    plan = parse_json_model(f"Sure!\n{payload}\nDone.", Plan)
    assert plan.tasks == [TaskModel(id="1", description="do it", files=["a.py"], dependencies=[])]
    assert plan.rationale == "one step"


def test_schema_mismatch_names_the_model_and_the_text() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_json_model('{"oops": true}', Plan)
    message = str(excinfo.value)
    assert "Plan" in message
    assert "oops" in message


def test_schema_mismatch_uses_the_offending_response_not_the_prompt() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_json_model('{"tasks": [], "rationale": ""}', Verdict)
    assert "rationale" in str(excinfo.value)


def test_broken_json_raises_value_error() -> None:
    with pytest.raises(ValueError):
        parse_json_model('{"passed": }', Verdict)


def test_no_json_raises_value_error() -> None:
    with pytest.raises(ValueError):
        parse_json_model("sorry, no JSON today", Verdict)
