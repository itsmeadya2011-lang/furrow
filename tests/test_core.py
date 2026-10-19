import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_extract_tool_calls_single():
    text = '{"tool": "read_file", "args": {"path": "src/foo.py"}}'
    calls = LLMClient.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0] == {"tool": "read_file", "args": {"path": "src/foo.py"}}


def test_extract_tool_calls_multiple():
    text = '{"tool": "list_files", "args": {"path": "."}} {\"tool\": \"done\", \"args\": {\"summary\": "done"}}'
    calls = LLMClient.extract_tool_calls(text)
    assert [c["tool"] for c in calls] == ["list_files", "done"]


def test_extract_tool_calls_tolerates_prose():
    text = (
        "Here is the call: {\"tool\": \"write_file\", \"args\": {\"path\": \"a.py\", \"content\": \"x\"}}\n"
        "Now done: {\"tool\": \"done\", \"args\": {\"summary\": \"wrote a.py\"}}"
    )
    calls = LLMClient.extract_tool_calls(text)
    assert [c["tool"] for c in calls] == ["write_file", "done"]


def test_extract_tool_calls_ignores_non_tool_json():
    text = '{"foo": "bar"} {"tool": "done", "args": {}}'
    calls = LLMClient.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "done"


def test_extract_tool_calls_empty():
    assert LLMClient.extract_tool_calls("") == []
    assert LLMClient.extract_tool_calls("no json here") == []
