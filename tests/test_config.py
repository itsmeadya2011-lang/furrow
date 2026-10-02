"""Tests for config models and Settings."""

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TaskStatus, TestResult


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="do something")
        assert t.id == "1"
        assert t.description == "do something"
        assert t.files == []
        assert t.dependencies == []
        assert t.status == TaskStatus.PENDING
        assert t.result is None

    def test_custom_fields(self):
        t = TaskModel(
            id="2",
            description="write tests",
            files=["tests/test_x.py"],
            dependencies=["1"],
            status=TaskStatus.COMPLETED,
            result="Done",
        )
        assert t.files == ["tests/test_x.py"]
        assert t.dependencies == ["1"]
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "Done"

    def test_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


class TestPlan:
    def test_valid_plan(self):
        p = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="Break into tasks",
        )
        assert len(p.tasks) == 1
        assert p.tasks[0].description == "task 1"
        assert p.rationale == "Break into tasks"

    def test_empty_plan(self):
        p = Plan(tasks=[], rationale="Already done")
        assert p.tasks == []

    def test_plan_from_dict(self):
        data = {
            "tasks": [
                {"id": "1", "description": "first", "files": [], "dependencies": []},
                {"id": "2", "description": "second", "files": ["b.py"], "dependencies": ["1"]},
            ],
            "rationale": "sequential tasks",
        }
        p = Plan(**data)
        assert len(p.tasks) == 2
        assert p.tasks[1].files == ["b.py"]


class TestTestResult:
    def test_passed_result(self):
        t = TestResult(passed=True, summary="All tests passed")
        assert t.passed is True
        assert t.failures == []

    def test_failed_result_with_failures(self):
        t = TestResult(passed=False, summary="3 failures", failures=["err1", "err2", "err3"])
        assert t.passed is False
        assert len(t.failures) == 3


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.llm_timeout == 120
        assert s.max_retries == 3

    def test_workspace_default_is_cwd(self):
        s = Settings()
        import os
        assert str(s.workspace) == os.getcwd()

    def test_provider_env_override(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        s = Settings()
        assert s.provider == Provider.OPENAI

    def test_max_cycles_env_override(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
        s = Settings()
        assert s.max_cycles == 10

    def test_max_parallel_env_override(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "2")
        s = Settings()
        assert s.max_parallel_tasks == 2

    def test_invalid_provider_raises(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "invalid")
        with pytest.raises(ValidationError):
            Settings()
