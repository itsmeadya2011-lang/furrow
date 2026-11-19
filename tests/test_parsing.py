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


# --- strip_code_fences edge cases ---


def test_strip_code_fences_multiple_fenced_blocks():
    text = 'intro ```json\n{"a": 1}\n```\nmiddle ```python\nprint(1)\n```\noutro'
    result = parsing.strip_code_fences(text)
    assert result == 'intro {"a": 1}\nmiddle print(1)\noutro'


def test_strip_code_fences_no_fences_unchanged():
    text = "plain text with no fences at all"
    assert parsing.strip_code_fences(text) == text


def test_strip_code_fences_empty_string():
    assert parsing.strip_code_fences("") == ""


def test_strip_code_fences_non_string_input_stringified():
    assert parsing.strip_code_fences({"a": 1}) == str({"a": 1})
    assert parsing.strip_code_fences(123) == "123"


# --- extract_json edge cases ---


def test_extract_json_nested_objects():
    text = 'outer {"level1": {"level2": {"level3": "deep"}}} tail'
    assert parsing.extract_json(text) == '{"level1": {"level2": {"level3": "deep"}}}'


def test_extract_json_empty_string():
    with pytest.raises(ValueError):
        parsing.extract_json("")


def test_extract_json_non_string_input():
    assert parsing.extract_json([1, 2, 3]) == "[1, 2, 3]"


# --- parse_json edge cases ---


def test_parse_json_deeply_nested_valid_json():
    text = '{"a": {"b": {"c": {"d": 1}}}}'
    assert parsing.parse_json(text) == {"a": {"b": {"c": {"d": 1}}}}


def test_parse_json_multiple_candidates_first_invalid_second_valid():
    text = '{bad json} then {"valid": true}'
    assert parsing.parse_json(text) == {"valid": True}


def test_parse_json_non_string_input():
    assert parsing.parse_json([1, 2, 3]) == [1, 2, 3]


# --- parse_plan edge cases ---


def test_parse_plan_task_with_extra_fields_ignored():
    text = (
        '{"tasks": [{"id": "t1", "description": "do it", "secret_field": "ignored"}], '
        '"rationale": "ok"}'
    )
    plan = parsing.parse_plan(text)
    assert plan.tasks[0].id == "t1"
    assert plan.tasks[0].description == "do it"


def test_parse_plan_non_string_input():
    with pytest.raises(ValueError):
        parsing.parse_plan({"tasks": []})


# --- parse_test_result edge cases ---


def test_parse_test_result_non_string_input():
    r = parsing.parse_test_result(12345)
    assert isinstance(r, TestResult)


def test_parse_test_result_none_input():
    r = parsing.parse_test_result(None)
    assert isinstance(r, TestResult)


def test_parse_test_result_json_with_extra_fields():
    r = parsing.parse_test_result(
        '{"passed": true, "summary": "ok", "failures": [], "extra_field": "ignored"}'
    )
    assert r.passed is True
    assert r.summary == "ok"


def test_parse_test_result_mixed_case_passed_failed():
    r = parsing.parse_test_result("Passed: true")
    assert r.passed is True
    r = parsing.parse_test_result("FAILED: true")
    assert r.passed is False


def test_parse_test_result_multiple_failure_lines_capped_at_ten():
    lines = "\n".join(f"error number {i}" for i in range(15))
    r = parsing.parse_test_result(lines)
    assert len(r.failures) == 10


# --- _iter_json_candidates edge cases ---


def test_iter_json_candidates_empty_string():
    assert list(parsing._iter_json_candidates("")) == []


def test_iter_json_candidates_non_string_input():
    assert list(parsing._iter_json_candidates([1, 2])) == ["[1, 2]"]
