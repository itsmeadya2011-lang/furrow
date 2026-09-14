import os
from pathlib import Path

import pytest

from furrow.config import Provider, Plan, Settings, TaskModel, TestResult


class TestProviderEnum:
    def test_provider_anthropic_value(self):
        assert Provider.ANTHROPIC.value == "anthropic"

    def test_provider_openai_value(self):
        assert Provider.OPENAI.value == "openai"

    def test_provider_ollama_value(self):
        assert Provider.OLLAMA.value == "ollama"

    def test_provider_is_str_enum(self):
        assert isinstance(Provider.ANTHROPIC, str)
        assert Provider.ANTHROPIC == "anthropic"


class TestTaskModel:
    def test_task_model_minimal(self):
        task = TaskModel(id="1", description="do thing")
        assert task.id == "1"
        assert task.description == "do thing"
        assert task.files == []
        assert task.dependencies == []
        assert task.status == "pending"
        assert task.result is None

    def test_task_model_with_all_fields(self):
        task = TaskModel(
            id="2",
            description="build api",
            files=["src/api.py"],
            dependencies=["1"],
            status="in_progress",
            result="done",
        )
        assert task.files == ["src/api.py"]
        assert task.dependencies == ["1"]
        assert task.status == "in_progress"
        assert task.result == "done"

    def test_task_model_default_files_is_independent(self):
        task1 = TaskModel(id="1", description="a")
        task2 = TaskModel(id="2", description="b")
        task1.files.append("x")
        assert task2.files == []


class TestPlan:
    def test_plan_creation(self):
        tasks = [TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")]
        plan = Plan(tasks=tasks, rationale="test rationale")
        assert len(plan.tasks) == 2
        assert plan.rationale == "test rationale"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="no tasks")
        assert plan.tasks == []
        assert plan.rationale == "no tasks"


class TestTestResult:
    def test_test_result_passed(self):
        result = TestResult(passed=True, summary="all good")
        assert result.passed is True
        assert result.summary == "all good"
        assert result.failures == []

    def test_test_result_failed_with_failures(self):
        result = TestResult(passed=False, summary="failures", failures=["error 1", "error 2"])
        assert result.passed is False
        assert result.failures == ["error 1", "error 2"]

    def test_test_result_failures_default_empty(self):
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []


class TestSettingsDefaults:
    def test_provider_default(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_model_default(self):
        s = Settings()
        assert s.model == "claude-sonnet-4-20250514"

    def test_planner_model_default(self):
        s = Settings()
        assert s.planner_model == "claude-3-5-haiku-20241022"

    def test_worker_model_default(self):
        s = Settings()
        assert s.worker_model == "claude-3-5-sonnet-20241022"

    def test_tester_model_default(self):
        s = Settings()
        assert s.tester_model == "claude-3-5-sonnet-20241022"

    def test_max_parallel_tasks_default(self):
        s = Settings()
        assert s.max_parallel_tasks == 5

    def test_max_cycles_default(self):
        s = Settings()
        assert s.max_cycles == 0

    def test_ollama_base_url_default(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_log_level_default(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_log_format_default(self):
        s = Settings()
        assert s.log_format == "console"

    def test_test_timeout_default(self):
        s = Settings()
        assert s.test_timeout == 120

    def test_workspace_default_is_path(self):
        s = Settings()
        assert isinstance(s.workspace, Path)

    def test_api_keys_default_none(self):
        s = Settings()
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None


class TestSettingsFromEnv:
    def test_max_parallel_tasks_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
        s = Settings()
        assert s.max_parallel_tasks == 10

    def test_max_cycles_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_CYCLES", "3")
        s = Settings()
        assert s.max_cycles == 3

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        s = Settings()
        assert s.provider == Provider.OPENAI

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_MODEL", "gpt-4")
        s = Settings()
        assert s.model == "gpt-4"

    def test_test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_TEST_TIMEOUT", "60")
        s = Settings()
        assert s.test_timeout == 60

    def test_ollama_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_OLLAMA_BASE_URL", "http://custom:1234")
        s = Settings()
        assert s.ollama_base_url == "http://custom:1234"
