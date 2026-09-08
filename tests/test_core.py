import pytest
import tempfile
from pathlib import Path
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.id == "1"
    assert task.description == "test task"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_values():
    task = TaskModel(
        id="2",
        description="another task",
        files=["file1.py", "file2.py"],
        dependencies=["1"],
        status="completed",
        result="done",
    )
    assert task.files == ["file1.py", "file2.py"]
    assert task.dependencies == ["1"]
    assert task.status == "completed"
    assert task.result == "done"


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0


def test_settings_custom_values():
    settings = Settings(
        provider=Provider.OPENAI,
        model="gpt-4",
        max_parallel_tasks=10,
        max_cycles=5,
    )
    assert settings.provider == Provider.OPENAI
    assert settings.model == "gpt-4"
    assert settings.max_parallel_tasks == 10
    assert settings.max_cycles == 5


def test_plan_with_empty_tasks():
    plan = Plan(tasks=[], rationale="no tasks needed")
    assert plan.tasks == []
    assert plan.rationale == "no tasks needed"


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_llm_client_list_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.py").write_text("world")

        client = LLMClient()
        files = client.list_files(tmpdir)

        assert "file1.txt" in files
        assert "subdir/file2.py" in files


def test_orchestrator_is_done_empty():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed():
    orchestrator = Orchestrator(goal="test")

    class MockTask:
        def __init__(self, status):
            self.status = status

    orchestrator._get_tasks = lambda: [
        MockTask("completed"),
        MockTask("completed"),
    ]
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_failed():
    orchestrator = Orchestrator(goal="test")

    class MockTask:
        def __init__(self, status):
            self.status = status

    orchestrator._get_tasks = lambda: [
        MockTask("completed"),
        MockTask("failed"),
    ]
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_pending():
    orchestrator = Orchestrator(goal="test")

    class MockTask:
        def __init__(self, status):
            self.status = status

    orchestrator._get_tasks = lambda: [
        MockTask("completed"),
        MockTask("pending"),
    ]
    assert orchestrator._is_done() is False