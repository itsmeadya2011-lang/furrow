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


def test_parse_plan_raises_on_garbage():
    with pytest.raises(ValueError):
        parsing.parse_plan("garbage")


def test_parse_plan_raises_on_tasks_not_a_list():
    with pytest.raises(ValueError):
        parsing.parse_plan('{"tasks": "not-a-list"}')


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


def test_strip_code_fences_single_fenced_block_returns_inner_content():
    text = "```json\n{\"key\": \"value\"}\n```"
    result = parsing.strip_code_fences(text)
    # The regex captures the content between the opening fence line and the
    # closing fence, which includes the trailing newline before the closing
    # fence. The inner content is preserved verbatim.
    assert result.strip() == '{"key": "value"}'
    assert "```" not in result


def test_strip_code_fences_single_block_without_info_string():
    text = "```\nplain text\n```"
    result = parsing.strip_code_fences(text)
    assert result.strip() == "plain text"
    assert "```" not in result


def test_strip_code_fences_multiple_blocks_strips_fence_lines():
    text = "intro\n```json\n{\"a\": 1}\n```\nmiddle\n```\n{\"b\": 2}\n```\noutro"
    result = parsing.strip_code_fences(text)
    assert "```" not in result
    assert "intro" in result
    assert "middle" in result
    assert "outro" in result
    assert '{"a": 1}' in result
    assert '{"b": 2}' in result


def test_strip_code_fences_no_fences_returns_text_unchanged():
    text = "just plain prose"
    assert parsing.strip_code_fences(text) == text


def test_strip_code_fences_empty_string():
    assert parsing.strip_code_fences("") == ""


def test_strip_code_fences_non_string_input_is_converted():
    assert parsing.strip_code_fences(None) == "None"
    assert parsing.strip_code_fences(42) == "42"


def test_strip_code_fences_unclosed_fence_strips_opening_line():
    text = "```json\n{\"key\": \"value\"}"
    result = parsing.strip_code_fences(text)
    assert "```" not in result
    assert '{"key": "value"}' in result


def test_strip_code_fences_fence_at_start_of_line_only():
    text = "```\nhello\n```"
    result = parsing.strip_code_fences(text)
    # The regex captures content up to the closing fence, including the trailing
    # newline before the closing fence.
    assert result.strip() == "hello"
    assert "```" not in result


def test_strip_code_fences_multiple_blocks_with_prose():
    text = "Here is one:\n```json\n{\"a\": 1}\n```\nAnd another:\n```\n{\"b\": 2}\n```"
    result = parsing.strip_code_fences(text)
    assert "```" not in result
    assert "Here is one:" in result
    assert "And another:" in result
    assert '{"a": 1}' in result
    assert '{"b": 2}' in result
