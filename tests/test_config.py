from __future__ import annotations

from furrow.config import Plan, Settings, TaskModel, TestResult


def test_task_model_defaults():
    t = TaskModel(id="1", description="do a thing")
    assert t.id == "1"
    assert t.description == "do a thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_defaults():
    p = Plan(tasks=[TaskModel(id="1", description="x")], rationale="why")
    assert p.rationale == "why"
    assert len(p.tasks) == 1


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.passed is True
    assert t.failures == []
    # Should not be collected by pytest as a test class.
    assert TestResult.__test__ is False


def test_settings_defaults():
    s = Settings()
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.retry_max_attempts == 3
    assert s.state_file is None


def test_settings_override():
    s = Settings(max_parallel_tasks=2, retry_max_attempts=5, max_cycles=3)
    assert s.max_parallel_tasks == 2
    assert s.retry_max_attempts == 5
    assert s.max_cycles == 3
