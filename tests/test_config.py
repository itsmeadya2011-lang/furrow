from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import (
    Plan,
    Provider,
    RunResult,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    WorkerResult,
)


def test_task_model_defaults():
    t = TaskModel(id="1", description="d")
    assert t.status == "pending"
    assert t.attempts == 0
    assert t.error is None
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None
    assert t.started_at is None
    assert t.finished_at is None


def test_task_status_enum():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.RUNNING == "running"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
    assert TaskStatus.BLOCKED == "blocked"
    assert TaskStatus.SKIPPED == "skipped"


def test_worker_result_defaults():
    w = WorkerResult()
    assert w.summary == ""
    assert w.success is True
    assert w.files_changed == []
    assert w.error is None
    assert w.iterations == 0


def test_run_result_defaults():
    r = RunResult()
    assert r.passed is False
    assert r.cycles == 0
    assert r.tasks_completed == 0
    assert r.tasks_failed == 0
    assert r.summary == ""
    assert r.stopped_reason == ""


def test_plan_and_test_result_still_work():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"

    tr = TestResult(passed=True, summary="ok", failures=[])
    assert tr.passed is True
    assert tr.summary == "ok"
    assert tr.failures == []


def test_settings_state_path_absolute_workspace():
    s = Settings()
    assert s.state_path(Path("/tmp/ws")) == Path("/tmp/ws/.furrow/state.json")


def test_settings_state_path_custom_state_file():
    s = Settings(state_file=Path("custom/state.json"))
    assert s.state_path(Path("/tmp/ws")) == Path("/tmp/ws/custom/state.json")


def test_settings_state_path_no_argument_uses_workspace():
    s = Settings(workspace=Path("/my/workspace"))
    assert s.state_path() == Path("/my/workspace/.furrow/state.json")


def test_test_result_not_collected_by_pytest():
    assert TestResult.__test__ is False
    assert set(TestResult.model_fields) == {"passed", "summary", "failures"}


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    s = Settings(_env_file=None)
    assert s.max_cycles == 7


def test_settings_provider_default():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
