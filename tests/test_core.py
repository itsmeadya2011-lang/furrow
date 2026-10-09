import pytest
from pathlib import Path
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_validation():
    # Valid status
    task = TaskModel(id="1", description="test", status="pending")
    assert task.status == "pending"

    task = TaskModel(id="1", description="test", status="completed")
    assert task.status == "completed"

    task = TaskModel(id="1", description="test", status="failed")
    assert task.status == "failed"

    # Invalid status should raise
    with pytest.raises(ValueError):
        TaskModel(id="1", description="test", status="invalid")


def test_plan_max_tasks():
    tasks = [TaskModel(id=str(i), description=f"task {i}") for i in range(6)]
    with pytest.raises(ValueError):
        Plan(tasks=tasks, rationale="too many")


def test_settings_validation():
    # Valid settings
    s = Settings(max_parallel_tasks=5, log_level="INFO")
    assert s.max_parallel_tasks == 5
    assert s.log_level == "INFO"

    # Invalid max_parallel_tasks
    with pytest.raises(ValueError):
        Settings(max_parallel_tasks=0)

    with pytest.raises(ValueError):
        Settings(max_parallel_tasks=21)

    # Invalid log_level
    with pytest.raises(ValueError):
        Settings(log_level="INVALID")

    # Valid log levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        s = Settings(log_level=level)
        assert s.log_level == level.upper()


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()


def test_test_result_defaults():
    t = TestResult(passed=False, summary="failed")
    assert t.failures == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
