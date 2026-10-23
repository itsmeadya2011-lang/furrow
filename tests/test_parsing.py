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


def test_strip_code_fences_single_fenced_block():
    """strip_code_fences returns the inner content of a single fenced block."""
    text = "```json\n{\"a\": 1}\n```"
    result = parsing.strip_code_fences(text)
    assert "1" in result
    assert "```" not in result


def test_strip_code_fences_plain_text():
    """strip_code_fences returns the text unchanged when there are no fences."""
    text = "just plain text"
    assert parsing.strip_code_fences(text) == "just plain text"


def test_strip_code_fences_multiple_blocks_strips_all():
    """strip_code_fences strips fence lines from text with multiple blocks."""
    text = "before\n```json\n{\"a\": 1}\n```\nmiddle\n```\n{\"b\": 2}\n```\nafter"
    result = parsing.strip_code_fences(text)
    assert "before" in result
    assert "after" in result
    assert "```" not in result


def test_strip_code_fences_non_string_input():
    """strip_code_fences coerces non-string input to a string."""
    assert parsing.strip_code_fences(None) == "None"
    assert parsing.strip_code_fences(42) == "42"


def test_extract_json_returns_first_balanced_object():
    """extract_json returns the first balanced JSON object in the text."""
    text = 'prefix {"a": 1} suffix {"b": 2}'
    result = parsing.extract_json(text)
    assert result == '{"a": 1}'


def test_extract_json_returns_first_balanced_array():
    """extract_json returns the first balanced JSON array in the text."""
    text = 'prefix [1, 2, 3] suffix [4, 5, 6]'
    result = parsing.extract_json(text)
    assert result == '[1, 2, 3]'


def test_extract_json_handles_nested_objects():
    """extract_json handles nested objects and arrays."""
    text = '{"outer": {"inner": [1, 2, {"deep": true}]}}'
    result = parsing.extract_json(text)
    assert result == text


def test_extract_json_handles_strings_with_braces():
    """extract_json ignores braces inside JSON string literals."""
    text = '{"key": "value with {braces} inside"}'
    result = parsing.extract_json(text)
    assert result == text


def test_extract_json_handles_escaped_quotes():
    """extract_json handles escaped quotes inside strings."""
    text = '{"key": "value with \\"escaped\\" quotes"}'
    result = parsing.extract_json(text)
    assert result == text


def test_parse_json_returns_parsed_value():
    """parse_json returns the parsed JSON value."""
    result = parsing.parse_json('{"a": 1, "b": [2, 3]}')
    assert result == {"a": 1, "b": [2, 3]}


def test_parse_json_returns_array():
    """parse_json can parse a JSON array."""
    result = parsing.parse_json('[1, 2, 3]')
    assert result == [1, 2, 3]


def test_parse_plan_extracts_from_fenced_block():
    """parse_plan extracts JSON from a fenced code block."""
    text = '```json\n{"tasks": [{"id": "t1", "description": "d"}], "rationale": "r"}\n```'
    plan = parsing.parse_plan(text)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "t1"


def test_parse_plan_extracts_from_prose():
    """parse_plan extracts JSON from prose text."""
    text = ('Here is the plan: '
            '{"tasks": [{"id": "t1", "description": "d"}], "rationale": "r"} thanks')
    plan = parsing.parse_plan(text)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "t1"


def test_parse_plan_includes_excerpt_on_error():
    """parse_plan includes an excerpt of the original text in the error message."""
    try:
        parsing.parse_plan("not valid json at all")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "not valid json at all" in str(e)


def test_parse_test_result_with_skipped():
    """parse_test_result handles a skipped test result."""
    r = parsing.parse_test_result(
        '{"passed": true, "summary": "skipped", "failures": [], "skipped": true}'
    )
    assert r.passed is True
    assert r.skipped is True


def test_parse_test_result_with_failures_array():
    """parse_test_result extracts failures from the JSON."""
    r = parsing.parse_test_result(
        '{"passed": false, "summary": "fail", "failures": ["err1", "err2"]}'
    )
    assert r.passed is False
    assert len(r.failures) == 2
    assert "err1" in r.failures
    assert "err2" in r.failures


def test_parse_test_result_heuristic_error_keyword():
    """parse_test_result detects failures from error keywords in prose."""
    r = parsing.parse_test_result("Traceback (most recent call last):")
    assert r.passed is False


def test_parse_test_result_heuristic_no_failures():
    """parse_test_result detects success from 'no failures' phrase."""
    r = parsing.parse_test_result("no failures detected")
    assert r.passed is True


def test_parse_test_result_never_raises_on_garbage():
    """parse_test_result never raises, even on garbage input."""
    r = parsing.parse_test_result("!!!@@@###")
    assert isinstance(r, TestResult)


def test_parse_test_result_never_raises_on_none():
    """parse_test_result never raises on None input."""
    r = parsing.parse_test_result(None)
    assert isinstance(r, TestResult)
