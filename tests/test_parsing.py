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


def test_parse_test_result_heuristic_explicit_prose_verdict():
    """An unquoted prose verdict is honoured, including the negative spelling.

    ``passed: false`` used to be read as a PASS because only the positive
    ``passed: true`` was recognised.
    """
    assert parsing.parse_test_result("passed: false").passed is False
    assert parsing.parse_test_result("passed=false").passed is False
    assert parsing.parse_test_result("passed: no").passed is False
    assert parsing.parse_test_result("passed: true").passed is True
    assert parsing.parse_test_result("passed: yes").passed is True
    assert parsing.parse_test_result("failed: true").passed is False
    assert parsing.parse_test_result("result: failed").passed is False


def test_parse_test_result_heuristic_zero_error_counters_are_not_failures():
    """Green zero-count summaries are passes and list no failures.

    ``error count is 0`` used to be read as a failure because the bare word
    ``error`` was matched, and a listed failure turns a pass into a failure.
    """
    for text in ["0 errors", "no failures", "5 passed, 0 failed", "1 passed, 0 errors"]:
        r = parsing.parse_test_result(text)
        assert r.passed is True, f"{text!r} should pass"
        assert r.failures == [], f"{text!r} should list no failures"

    # A single real failure alongside a zero counter is still a failure.
    r = parsing.parse_test_result("1 failed, 0 errors")
    assert r.passed is False
    assert r.failures, "the failing line must still be reported"


def test_extract_json_returns_the_first_balanced_candidate():
    assert parsing.extract_json("prefix [draft] suffix") == "[draft]"


def test_parse_json_skips_a_non_json_bracket_before_the_object():
    plan = parsing.parse_plan(
        'Here is [draft] the plan: {"tasks": [{"id": "t1", "description": "x"}], "rationale": "r"}'
    )
    assert plan.tasks[0].id == "t1"


def test_parse_json_skips_a_non_json_brace_before_the_object():
    plan = parsing.parse_plan('I considered {a} and chose {"tasks": [], "rationale": "ok"}')
    assert plan.rationale == "ok"


def test_parse_plan_skips_a_leading_json_array_before_the_object():
    plan = parsing.parse_plan('[1, 2, 3] then {"tasks": [], "rationale": "ok"}')
    assert plan.rationale == "ok"


def test_parse_json_returns_the_first_valid_candidate():
    assert parsing.parse_json('noise [x] then {"a": 1}') == {"a": 1}


def test_parse_json_raises_when_no_candidate_is_valid_json():
    with pytest.raises(ValueError):
        parsing.parse_json("[draft] {a}")


def test_parse_plan_finds_a_valid_object_nested_in_an_invalid_one():
    plan = parsing.parse_plan('{not json {"tasks": [], "rationale": "ok"}}')
    assert plan.rationale == "ok"


def test_parse_test_result_uses_a_json_object_after_stray_brackets():
    tr = parsing.parse_test_result('noise [x] {"passed": false, "summary": "s", "failures": ["a"]}')
    assert tr.passed is False
    assert tr.failures == ["a"]


def _raise_recursionerror(_text: str):
    raise RecursionError("maximum recursion depth exceeded")


def test_parse_json_converts_recursionerror_to_valueerror(monkeypatch):
    # A deeply nested payload makes ``json.loads`` raise RecursionError; the
    # parser must surface it as the documented ValueError instead.
    monkeypatch.setattr(parsing.json, "loads", _raise_recursionerror)
    with pytest.raises(ValueError):
        parsing.parse_json("[[1]]")


def test_parse_plan_converts_recursionerror_to_valueerror(monkeypatch):
    monkeypatch.setattr(parsing.json, "loads", _raise_recursionerror)
    with pytest.raises(ValueError):
        parsing.parse_plan("[[1]]")


def test_parse_test_result_falls_back_on_recursionerror(monkeypatch):
    monkeypatch.setattr(parsing.json, "loads", _raise_recursionerror)
    r = parsing.parse_test_result("[[1]]")
    assert isinstance(r, TestResult)
