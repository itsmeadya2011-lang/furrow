"""Tests for furrow configuration models and settings."""

import pytest
from pydantic import ValidationError

from furrow.config import (
    OrchestratorState,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    setup_logging,
)


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_model_defaults():
    t = TaskModel(id="1", description="test task")
    assert t.status == TaskStatus.PENDING.value
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_custom_status():
    t = TaskModel(id="1", description="test", status="in_progress")
    assert t.status == "in_progress"


def test_task_model_with_files():
    t = TaskModel(
        id="1",
        description="do thing",
        files=["src/auth.py", "tests/test_auth.py"],
        dependencies=["2"],
    )
    assert t.files == ["src/auth.py", "tests/test_auth.py"]
    assert t.dependencies == ["2"]


def test_plan_parse():
    p = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_plan_from_dict():
    p = Plan(
        tasks=[{"id": "1", "description": "test", "files": [], "dependencies": []}],
        rationale="test rationale",
    )
    assert len(p.tasks) == 1
    assert p.tasks[0].description == "test"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="1 test failed", failures=["test_auth.test_login"])
    assert t.passed is False
    assert len(t.failures) == 1


def test_task_status_enum_values():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.IN_PROGRESS == "in_progress"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"


def test_orchestrator_state_defaults():
    state = OrchestratorState(goal="test goal")
    assert state.goal == "test goal"
    assert state.cycle == 0
    assert state.task_history == []
    assert state.test_history == []
    assert state.completed is False


def test_orchestrator_state_add_task():
    state = OrchestratorState(goal="test")
    task = TaskModel(id="1", description="task 1")
    state.add_task(task)
    assert len(state.task_history) == 1

    # Adding a task with same id should replace
    task.result = "done"
    state.add_task(task)
    assert len(state.task_history) == 1
    assert state.task_history[0].result == "done"


def test_orchestrator_state_update_task():
    state = OrchestratorState(goal="test")
    task = TaskModel(id="1", description="task 1")
    state.update_task(task)
    assert len(state.task_history) == 1

    task.status = TaskStatus.COMPLETED.value
    state.update_task(task)
    assert len(state.task_history) == 1
    assert state.task_history[0].status == "completed"


def test_orchestrator_state_json_roundtrip():
    state = OrchestratorState(goal="test goal")
    state.cycle = 3
    state.task_history = [TaskModel(id="1", description="task 1", status="completed")]

    json_str = state.model_dump_json()
    restored = OrchestratorState.model_validate_json(json_str)

    assert restored.goal == "test goal"
    assert restored.cycle == 3
    assert len(restored.task_history) == 1
    assert restored.task_history[0].description == "task 1"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_max_parallel_minimum():
    """max_parallel_tasks < 1 should be clamped to 1."""
    s = Settings(max_parallel_tasks=0)
    assert s.max_parallel_tasks == 1


def test_settings_log_level_validation():
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"


def test_settings_invalid_log_level():
    with pytest.raises(ValidationError):
        Settings(log_level="INVALID")


def test_settings_state_file_default():
    s = Settings()
    assert s.state_file == ".furrow-state.json"


def test_setup_logging():
    """Ensure setup_logging doesn't raise."""
    setup_logging("INFO")
