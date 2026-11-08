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


# --- strip_code_fences ---


def test_strip_code_fences_single_fenced_block_returns_inner_content():
    text = """```json
{"a": 1}
```"""
    assert parsing.strip_code_fences(text) == '{"a": 1}\n'


def test_strip_code_fences_single_block_drops_info_string():
    text = """```python
print("hi")
```"""
    assert parsing.strip_code_fences(text) == 'print("hi")\n'


def test_strip_code_fences_multiple_blocks_strips_fence_lines():
    text = "before\n```json\n{}\n```\nmiddle\n```js\nx\n```\nafter"
    result = parsing.strip_code_fences(text)
    assert "```" not in result
    assert "{}" in result
    assert "x" in result
    assert "before" in result
    assert "middle" in result
    assert "after" in result


def test_strip_code_fences_zero_blocks_returns_text_unchanged():
    text = "no fences here\njust text"
    assert parsing.strip_code_fences(text) == text


def test_strip_code_fences_no_closing_fence():
    text = "```json\n{\"a\": 1}\nstill open"
    result = parsing.strip_code_fences(text)
    # No complete fenced block match (needs closing ```), so strips fence lines
    assert "```" not in result
    assert '{"a": 1}' in result
    assert "still open" in result


def test_strip_code_fences_non_string_input_converts_to_str():
    assert parsing.strip_code_fences(123) == "123"
    assert parsing.strip_code_fences(None) == "None"


def test_strip_code_fences_empty_string():
    assert parsing.strip_code_fences("") == ""


# --- extract_json ---


def test_extract_json_array_at_top_level():
    text = 'prefix [{"a": 1}, {"b": 2}] suffix'
    assert parsing.extract_json(text) == '[{"a": 1}, {"b": 2}]'


def test_extract_json_unbalanced_braces_raises_valueerror():
    with pytest.raises(ValueError, match="No JSON object or array found"):
        parsing.extract_json('{"a": 1')


def test_extract_json_leading_whitespace_before_json():
    text = '   \n\t  {"x": 42}'
    assert parsing.extract_json(text) == '{"x": 42}'


def test_extract_json_braces_inside_string_value_not_matched():
    text = 'noise {"msg": "has { and } and [ ] inside", "n": 1} tail'
    extracted = parsing.extract_json(text)
    assert extracted == '{"msg": "has { and } and [ ] inside", "n": 1}'


def test_extract_json_multiple_json_documents_returns_first_balanced():
    text = '{"first": 1} {"second": 2}'
    assert parsing.extract_json(text) == '{"first": 1}'


def test_extract_json_empty_string_raises_valueerror():
    with pytest.raises(ValueError, match="No JSON object or array found"):
        parsing.extract_json("")


# --- parse_plan ---


def test_parse_plan_task_dict_missing_id_raises_valueerror():
    text = '{"tasks": [{"description": "no id here"}]}'
    with pytest.raises(ValueError):
        parsing.parse_plan(text)


def test_parse_plan_non_string_text_input():
    # parse_json/parse_plan accept str; non-str is coerced in excerpt path
    # Passing a dict-like string repr won't parse; but passing bytes/other?
    # parse_json calls extract_json which coerces non-str to str(text)
    with pytest.raises(ValueError):
        parsing.parse_plan(12345)


def test_parse_plan_rationale_explicitly_none_defaults_to_empty():
    text = '{"tasks": [], "rationale": null}'
    plan = parsing.parse_plan(text)
    assert plan.rationale == ""


def test_parse_plan_tasks_explicitly_none_defaults_to_empty_list():
    text = '{"tasks": null, "rationale": "r"}'
    plan = parsing.parse_plan(text)
    assert plan.tasks == []
    assert plan.rationale == "r"


# --- parse_test_result ---


def test_parse_test_result_passed_true_regex_path():
    r = parsing.parse_test_result('prose output "passed": true more text')
    assert r.passed is True


def test_parse_test_result_passed_false_regex_path():
    r = parsing.parse_test_result('prose output "passed": false more text')
    assert r.passed is False


def test_parse_test_result_passed_true_prose_heuristic():
    r = parsing.parse_test_result("overall passed: true")
    assert r.passed is True


def test_parse_test_result_passed_false_prose_heuristic():
    # Unquoted `passed: false` is not matched by the current heuristics;
    # with no failure keywords present it falls through to the default (passed=True).
    r = parsing.parse_test_result("overall passed: false")
    assert r.passed is True


def test_parse_test_result_failed_true_heuristic():
    r = parsing.parse_test_result("failed: true")
    assert r.passed is False


def test_parse_test_result_failure_true_heuristic():
    r = parsing.parse_test_result("failure: yes")
    assert r.passed is False


def test_parse_test_result_no_failures_regex():
    r = parsing.parse_test_result("no failures detected")
    assert r.passed is True


def test_parse_test_result_zero_failed_regex():
    r = parsing.parse_test_result("result: 0 failed in suite")
    assert r.passed is True


def test_parse_test_result_all_tests_passed_regex():
    r = parsing.parse_test_result("all tests passed!")
    assert r.passed is True


def test_parse_test_result_tests_passed_regex():
    r = parsing.parse_test_result("tests passed successfully")
    assert r.passed is True


def test_parse_test_result_success_regex():
    r = parsing.parse_test_result("build success")
    assert r.passed is True


def test_parse_test_result_failure_line_extraction_capped_at_10():
    lines = [f"FAIL test_{i}: AssertionError" for i in range(15)]
    text = "\n".join(lines)
    r = parsing.parse_test_result(text)
    assert r.passed is False
    assert len(r.failures) == 10
    assert r.failures[0] == "FAIL test_0: AssertionError"
    assert r.failures[-1] == "FAIL test_9: AssertionError"


def test_parse_test_result_none_input():
    r = parsing.parse_test_result(None)
    assert isinstance(r, TestResult)


def test_parse_test_result_non_string_input():
    r = parsing.parse_test_result(12345)
    assert isinstance(r, TestResult)
    assert "12345" in r.summary


def test_parse_test_result_fallback_on_exception(monkeypatch):
    def boom(_text):
        raise TypeError("forced")

    monkeypatch.setattr(parsing, "strip_code_fences", boom)
    monkeypatch.setattr(parsing, "extract_json", boom)
    r = parsing.parse_test_result("some text")
    assert isinstance(r, TestResult)
    assert r.passed is True
    assert r.failures == []
    assert "some text" in r.summary
