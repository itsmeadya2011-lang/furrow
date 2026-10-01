"""Tests for config models."""
from __future__ import annotations

from furrow.config import Plan, Provider, Settings, TaskModel, TaskStatus, TestResult


class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"


class TestTaskStatus:
    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestTaskModel:
    def test_default_status(self):
        t = TaskModel(id="1", description="do thing")
        assert t.status == TaskStatus.PENDING
        assert t.files == []
        assert t.dependencies == []
        assert t.result is None

    def test_with_explicit_status(self):
        t = TaskModel(id="1", description="do thing", status=TaskStatus.COMPLETED, result="done")
        assert t.status == TaskStatus.COMPLETED
        assert t.result == "done"


class TestPlan:
    def test_plan_creation(self):
        p = Plan(tasks=[TaskModel(id="1", description="task1")], rationale="reason")
        assert len(p.tasks) == 1
        assert p.rationale == "reason"

    def test_plan_empty(self):
        p = Plan(tasks=[], rationale="nothing to do")
        assert p.tasks == []


class TestTestResult:
    def test_passing_result(self):
        t = TestResult(passed=True, summary="all good", failures=[])
        assert t.passed is True
        assert t.failures == []

    def test_failing_result(self):
        t = TestResult(passed=False, summary="bad", failures=["test_1 failed"])
        assert t.passed is False
        assert len(t.failures) == 1


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_cycles == 0
        assert s.max_parallel_tasks == 5
        assert s.max_tokens == 8192
        assert s.timeout == 120.0
        assert s.max_retries == 3
        assert s.retry_delay == 2.0

    def test_provider_override(self):
        s = Settings(provider=Provider.OPENAI)
        assert s.provider == Provider.OPENAI

    def test_custom_values(self):
        s = Settings(max_cycles=5, max_parallel_tasks=3, max_tokens=4096)
        assert s.max_cycles == 5
        assert s.max_parallel_tasks == 3
        assert s.max_tokens == 4096
