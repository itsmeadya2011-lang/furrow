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


def test_parse_test_result_zero_count_lines_not_reported_as_failures():
    """Zero-count summary lines must not appear in the failures list."""
    text = "5 passed, 0 failed, 0 errors"
    r = parsing.parse_test_result(text)
    assert r.passed is True
    assert r.failures == []


def test_parse_test_result_failed_with_zero_counter_not_a_failure():
    text = "FAILED (failures=0, errors=0)"
    r = parsing.parse_test_result(text)
    assert r.passed is True
    assert r.failures == []


def test_parse_test_result_no_failures_phrase():
    r = parsing.parse_test_result("no failures detected")
    assert r.passed is True
    assert r.failures == []


def test_parse_test_result_real_failures_collected():
    text = (
        "FAILED tests/test_x.py::test_a - assertion error\n"
        "FAILED tests/test_y.py::test_b - timeout\n"
        "1 failed, 2 passed"
    )
    r = parsing.parse_test_result(text)
    assert r.passed is False
    assert len(r.failures) >= 2


def test_parse_test_result_failure_cap_at_ten():
    lines = "\n".join(f"FAILED tests/test_{i}.py::test_{i} - error" for i in range(15))
    r = parsing.parse_test_result(lines)
    assert len(r.failures) <= 10


def test_parse_test_result_none_input():
    r = parsing.parse_test_result(None)  # type: ignore[arg-type]
    assert isinstance(r, TestResult)


def test_parse_test_result_non_string_input():
    r = parsing.parse_test_result(12345)  # type: ignore[arg-type]
    assert isinstance(r, TestResult)


def test_parse_test_result_code_fenced_json():
    text = '```json\n{"passed": false, "summary": "x", "failures": ["a", "b"]}\n```'
    r = parsing.parse_test_result(text)
    assert r.passed is False
    assert r.failures == ["a", "b"]
