import pytest
from pydantic import ValidationError
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_requires_tasks_and_rationale():
    with pytest.raises(ValidationError):
        Plan(rationale="only rationale")
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="do thing")])


def test_test_result_failures_defaults_empty():
    t = TestResult(passed=False, summary="bad")
    assert t.failures == []


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch._get_tasks = lambda: tasks
    assert orch._is_done() is True


def test_orchestrator_is_done_some_failed():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orch._get_tasks = lambda: tasks
    assert orch._is_done() is False


def test_orchestrator_is_done_not_all_completed():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="pending"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    orch._get_tasks = lambda: tasks
    assert orch._is_done() is False


def test_orchestrator_get_tasks_returns_tasks():
    orch = Orchestrator(goal="test")
    tasks = [TaskModel(id="1", description="a")]
    orch._get_tasks = lambda: tasks
    assert orch._get_tasks() == tasks


async def test_llm_client_write_and_read_file(tmp_path):
    client = LLMClient()
    target = tmp_path / "subdir" / "test.txt"
    await client.write_file(target, "hello world")
    assert target.exists()
    content = await client.read_file(target)
    assert content == "hello world"


def test_llm_client_list_files(tmp_path):
    client = LLMClient()
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("c")
    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "b.txt" in files
    assert "sub/c.txt" in files
