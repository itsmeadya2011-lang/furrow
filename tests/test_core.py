"""Model and settings contract tests for :mod:`furrow.config`."""

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_requires_rationale():
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="do thing")])


def test_plan_requires_tasks():
    with pytest.raises(ValidationError):
        Plan(rationale="ok")


def test_plan_allows_empty_task_list():
    plan = Plan(tasks=[], rationale="nothing left to do")
    assert plan.tasks == []
    assert plan.rationale == "nothing left to do"


def test_task_model_defaults():
    task = TaskModel(id="1", description="do thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_mutable_defaults_are_not_shared():
    first = TaskModel(id="1", description="a")
    second = TaskModel(id="2", description="b")
    first.files.append("a.py")
    first.dependencies.append("1")
    assert second.files == []
    assert second.dependencies == []


def test_task_model_round_trips_dependencies():
    task = TaskModel(id="2", description="do the other thing", dependencies=["1", "3"])
    assert task.dependencies == ["1", "3"]
    assert task.model_dump()["dependencies"] == ["1", "3"]
    reloaded = TaskModel.model_validate(task.model_dump())
    assert reloaded.dependencies == ["1", "3"]


def test_task_model_round_trips_status_and_result():
    task = TaskModel(id="1", description="d", status="completed", result="did it")
    reloaded = TaskModel.model_validate(task.model_dump())
    assert reloaded.status == "completed"
    assert reloaded.result == "did it"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_defaults_failures_to_empty_list():
    t = TestResult(passed=False, summary="boom")
    assert t.failures == []
    assert t.passed is False
    assert t.summary == "boom"


def test_test_result_is_not_collected_by_pytest():
    # `TestResult` matches pytest's `Test*` collection rule, so it must opt out.
    assert TestResult.__test__ is False


def test_settings_reads_furrow_prefixed_env_vars(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "3")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "7")
    configured = Settings(_env_file=None)
    assert configured.max_cycles == 3
    assert configured.max_parallel_tasks == 7


def test_settings_ignores_unprefixed_env_vars(monkeypatch):
    monkeypatch.setenv("MAX_CYCLES", "11")
    assert Settings(_env_file=None).max_cycles != 11


def test_settings_constructor_args_win_over_env(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "9")
    assert Settings(_env_file=None, max_cycles=2).max_cycles == 2


def test_settings_is_constructible_for_tests():
    configured = Settings(_env_file=None, max_cycles=4, max_parallel_tasks=2)
    assert configured.max_cycles == 4
    assert configured.max_parallel_tasks == 2
    assert configured.planner_model
    assert configured.worker_model
    assert configured.tester_model