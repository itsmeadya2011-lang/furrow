from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestProvider:
    def test_provider_enum_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_provider_enum_is_str(self):
        assert isinstance(Provider.ANTHROPIC, str)
        assert Provider.ANTHROPIC == "anthropic"

    def test_provider_enum_members(self):
        members = list(Provider)
        assert len(members) == 3
        assert Provider.ANTHROPIC in members
        assert Provider.OPENAI in members
        assert Provider.OLLAMA in members


class TestSettings:
    def test_default_provider(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_default_model(self):
        s = Settings()
        assert s.model == "claude-sonnet-4-20250514"

    def test_default_planner_model(self):
        s = Settings()
        assert s.planner_model == "claude-3-5-haiku-20241022"

    def test_default_worker_model(self):
        s = Settings()
        assert s.worker_model == "claude-3-5-sonnet-20241022"

    def test_default_tester_model(self):
        s = Settings()
        assert s.tester_model == "claude-3-5-sonnet-20241022"

    def test_default_max_parallel_tasks(self):
        s = Settings()
        assert s.max_parallel_tasks == 5

    def test_default_max_cycles(self):
        s = Settings()
        assert s.max_cycles == 0

    def test_default_ollama_base_url(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_log_level(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_workspace(self):
        s = Settings()
        assert s.workspace == Path.cwd()

    def test_anthropic_provider(self):
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.provider == Provider.ANTHROPIC

    def test_openai_provider(self):
        s = Settings(provider=Provider.OPENAI)
        assert s.provider == Provider.OPENAI

    def test_ollama_provider(self):
        s = Settings(provider=Provider.OLLAMA)
        assert s.provider == Provider.OLLAMA

    def test_custom_values(self):
        s = Settings(
            provider=Provider.OPENAI,
            model="gpt-4",
            max_parallel_tasks=10,
            max_cycles=3,
            workspace=Path("/custom/path"),
        )
        assert s.provider == Provider.OPENAI
        assert s.model == "gpt-4"
        assert s.max_parallel_tasks == 10
        assert s.max_cycles == 3
        assert s.workspace == Path("/custom/path")

    def test_optional_api_keys_default_none(self):
        s = Settings()
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None


class TestTaskModel:
    def test_task_model_creation(self):
        t = TaskModel(id="1", description="do something")
        assert t.id == "1"
        assert t.description == "do something"
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_task_model_with_files(self):
        t = TaskModel(id="1", description="do something", files=["a.py", "b.py"])
        assert t.files == ["a.py", "b.py"]

    def test_task_model_with_dependencies(self):
        t = TaskModel(id="2", description="do something", dependencies=["1"])
        assert t.dependencies == ["1"]


class TestPlan:
    def test_plan_creation(self):
        tasks = [TaskModel(id="1", description="task")]
        p = Plan(tasks=tasks, rationale="test")
        assert len(p.tasks) == 1
        assert p.rationale == "test"


class TestTestResult:
    def test_test_result_passed(self):
        t = TestResult(passed=True, summary="ok")
        assert t.passed is True
        assert t.summary == "ok"
        assert t.failures == []

    def test_test_result_failed(self):
        t = TestResult(passed=False, summary="failed", failures=["error1"])
        assert t.passed is False
        assert t.summary == "failed"
        assert t.failures == ["error1"]
