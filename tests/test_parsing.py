from __future__ import annotations

import pytest

from furrow.agents import parsing
from furrow.config import Plan, TestResult


def test_fenced_json_block_parses():
    text = """```json
{"tasks": [{"id": "t1", "description": "do it"}], "rationale": "ok"}
```"""
    plan = parsing.parse_plan(text)
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1


def test_json_surrounded_by_prose_parses():
    text = (
        "Here is the plan:\n"
        '{"tasks": [{"id": "t1", "description": "x"}], "rationale": "r"}\n'
        "Let me know"
    )
    plan = parsing.parse_plan(text)
    assert isinstance(plan, Plan)
    assert plan.rationale == "r"


def test_nested_objects_and_arrays_parse():
    text = (
        '{"tasks": [{"id": "t1", "description": "x", "files": ["a.py"], "dependencies": ["t0"]}], '
        '"rationale": "hi"}'
    )
    plan = parsing.parse_plan(text)
    assert plan.tasks[0].files == ["a.py"]
    assert plan.tasks[0].dependencies == ["t0"]


def test_braces_inside_string_values_do_not_break_depth_scan():
    text = '{"tasks": [{"id": "t1", "description": "use { and } chars"}], "rationale": "ok"}'
    plan = parsing.parse_plan(text)
    assert "{" in plan.tasks[0].description


def test_escaped_quotes_inside_strings():
    text = '{"tasks": [{"id": "t1", "description": "say \\"hello\\""}], "rationale": "ok"}'
    plan = parsing.parse_plan(text)
    assert '\\"hello\\"' in plan.tasks[0].description or '"hello"' in plan.tasks[0].description


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        parsing.extract_json("no json here")


def test_parse_json_raises_valueerror_not_jsondecodeerror_on_malformed():
    with pytest.raises(ValueError):
        parsing.parse_json('{"bad": }')


def test_parse_plan_builds_plan_tolerates_missing_rationale():
    text = '{"tasks": [{"id": "t1", "description": "d"}]}'
    plan = parsing.parse_plan(text)
    assert isinstance(plan, Plan)
    assert plan.rationale == ""


def test_parse_plan_raises_on_tasks_not_a_list():
    with pytest.raises(ValueError):
        parsing.parse_plan('{"tasks": "not-a-list"}')


def test_parse_plan_raises_on_garbage():
    with pytest.raises(ValueError):
        parsing.parse_plan("garbage")


def test_strip_code_fences_returns_inner_block_for_single_fence():
    """A single fenced block yields its inner content, info string dropped."""
    text = "```json\n{\"tasks\": [], \"rationale\": \"x\"}\n```"
    assert parsing.strip_code_fences(text) == '{"tasks": [], "rationale": "x"}'


def test_strip_code_fences_plain_fence():
    """A plain `````` fence (no info string) is also unwrapped."""
    text = "```\n{\"tasks\": []}\n```"
    assert parsing.strip_code_fences(text) == '{"tasks": []}'


def test_strip_code_fences_multiple_blocks_keeps_remainder():
    """Multiple fenced blocks are unwrapped and the surrounding text kept."""
    text = "intro\n```json\n{\"a\": 1}\n```\noutro\n```\n{\"b\": 2}\n```"
    result = parsing.strip_code_fences(text)
    assert "intro" in result
    assert "outro" in result
    assert '{"a": 1}' in result
    assert '{"b": 2}' in result
    assert "```" not in result


def test_strip_code_fences_no_fences_returns_text_unchanged():
    text = '{"tasks": []}'
    assert parsing.strip_code_fences(text) == text


def test_strip_code_fences_non_string_input_is_stringified():
    assert parsing.strip_code_fences(None) == "None"  # type: ignore[arg-type]


def test_parse_plan_missing_tasks_defaults_to_empty_list():
    """A plan with no ``tasks`` key yields an empty task list."""
    plan = parsing.parse_plan('{"rationale": "nothing to do"}')
    assert plan.tasks == []


def test_parse_plan_error_message_includes_reason_and_excerpt():
    """A parse failure carries the underlying reason and a short excerpt."""
    text = "garbage"
    with pytest.raises(ValueError) as exc_info:
        parsing.parse_plan(text)
    message = str(exc_info.value)
    assert "Excerpt" in message
    assert text[:300] in message


def test_parse_plan_missing_rationale_defaults_to_empty_string():
    """A plan without ``rationale`` yields an empty rationale."""
    plan = parsing.parse_plan('{"tasks": [{"id": "t1", "description": "d"}]}')
    assert plan.rationale == ""


def test_parse_plan_none_rationale_defaults_to_empty_string():
    """An explicit ``null`` rationale is treated as an empty string."""
    plan = parsing.parse_plan('{"tasks": [], "rationale": null}')
    assert plan.rationale == ""


def test_parse_plan_none_tasks_defaults_to_empty_list():
    """An explicit ``null`` tasks value is treated as an empty list."""
    plan = parsing.parse_plan('{"tasks": null, "rationale": "r"}')
    assert plan.tasks == []


def test_parse_plan_normalises_unknown_task_status_to_pending():
    """An unrecognised status from untrusted planner output becomes ``pending``."""
    plan = parsing.parse_plan(
        '{"tasks": [{"id": "t1", "description": "d", "status": "wut"}], "rationale": "r"}'
    )
    assert plan.tasks[0].status == "pending"


def test_parse_test_result_json():
    r = parsing.parse_test_result('{"passed": false, "summary": "x", "failures": ["a"]}')
    assert isinstance(r, TestResult)
    assert r.passed is False
    assert r.failures == ["a"]


def test_parse_test_result_heuristic_zero_failed():
    r = parsing.parse_test_result("0 failed")
    assert r.passed is True


def test_parse_test_result_heuristic_two_failed_or_failed_tests():
    r = parsing.parse_test_result("2 failed")
    assert r.passed is False
    r = parsing.parse_test_result("FAILED tests")
    assert r.passed is False


def test_parse_test_result_heuristic_all_passed():
    r = parsing.parse_test_result("All tests passed")
    assert r.passed is True


def test_parse_test_result_heuristic_assertion_error():
    r = parsing.parse_test_result("AssertionError: boom")
    assert r.passed is False
    assert any("AssertionError" in f for f in r.failures)


def test_parse_test_result_never_raises_on_empty_string():
    r = parsing.parse_test_result("")
    assert isinstance(r, TestResult)
