"""Tests for furrow.config — Settings, TaskModel, Plan, Provider."""

from __future__ import annotations

import pytest

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)


class TestSettings:
    def test_settings_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify default Settings values."""
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.model == "claude-sonnet-4-20250514"
        assert s.planner_model == "claude-3-5-haiku-20241022"
        assert s.worker_model == "claude-3-5-sonnet-20241022"
        assert s.tester_model == "claude-3-5-sonnet-20241022"
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None
        assert s.ollama_base_url == "http://localhost:11434"
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.log_level == "INFO"

    def test_settings_anthropic_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify provider override via env var."""
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        s = Settings()
        assert s.provider == Provider.OPENAI


class TestTaskModel:
    def test_task_model_defaults(self) -> None:
        """Verify TaskModel default values."""
        t = TaskModel(id="1", description="do thing")
        assert t.id == "1"
        assert t.description == "do thing"
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_task_model_with_explicit_values(self) -> None:
        t = TaskModel(
            id="2",
            description="another",
            files=["src/x.py"],
            dependencies=["1"],
            status="completed",
            result="done",
        )
        assert t.files == ["src/x.py"]
        assert t.dependencies == ["1"]
        assert t.status == "completed"
        assert t.result == "done"


class TestPlan:
    def test_plan_validation(self) -> None:
        """Verify Plan model works."""
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="task one"),
                TaskModel(id="2", description="task two", dependencies=["1"]),
            ],
            rationale="build the thing",
        )
        assert len(plan.tasks) == 2
        assert plan.rationale == "build the thing"
        assert plan.tasks[0].id == "1"
        assert plan.tasks[1].dependencies == ["1"]

    def test_plan_model_dump(self) -> None:
        plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
        data = plan.model_dump()
        assert "tasks" in data
        assert "rationale" in data
        assert data["rationale"] == "r"


class TestTestResult:
    def test_test_result_defaults(self) -> None:
        t = TestResult(passed=True, summary="ok")
        assert t.passed is True
        assert t.summary == "ok"
        assert t.failures == []


class TestProviderEnum:
    def test_provider_enum_values(self) -> None:
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"

    def test_provider_enum_membership(self) -> None:
        assert Provider("anthropic") is Provider.ANTHROPIC
        assert Provider("openai") is Provider.OPENAI
        assert Provider("ollama") is Provider.OLLAMA