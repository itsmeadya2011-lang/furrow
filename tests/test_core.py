import pytest
from furrow.agents.planner import extract_json
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_extract_json_plain():
    raw = '{"key": "value"}'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_markdown_fence():
    raw = 'Here is the JSON:\n```json\n{"key": "value"}\n```\nThanks!'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_fence_no_language():
    raw = 'Some text\n```\n{"key": "value"}\n```\nMore text'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_with_text_around():
    raw = 'Sure! {"key": "value"} Let me know if you need anything else.'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json("No JSON here at all.")


def test_extract_json_malformed_raises():
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        extract_json('{"key": "value"')


def test_orchestrator_get_tasks_empty_without_plan():
    orch = Orchestrator(goal="test")
    assert orch._get_tasks() == []


def test_orchestrator_is_done_true_when_no_tasks():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is True


def test_orchestrator_is_done_true_when_all_completed():
    orch = Orchestrator(goal="test")
    orch._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert orch._is_done() is True


def test_orchestrator_is_done_false_when_failed():
    orch = Orchestrator(goal="test")
    orch._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert orch._is_done() is False


def test_orchestrator_is_done_false_when_pending():
    orch = Orchestrator(goal="test")
    orch._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="ok",
    )
    assert orch._is_done() is False
