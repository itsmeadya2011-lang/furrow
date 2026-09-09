import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestTaskModel:
    def test_task_model_defaults(self):
        task = TaskModel(id="1", description="test task")
        assert task.id == "1"
        assert task.description == "test task"
        assert task.files == []
        assert task.dependencies == []
        assert task.status == "pending"
        assert task.result is None

    def test_task_model_with_all_fields(self):
        task = TaskModel(
            id="2",
            description="complex task",
            files=["src/a.py", "src/b.py"],
            dependencies=["1"],
            status="completed",
            result="done",
        )
        assert task.files == ["src/a.py", "src/b.py"]
        assert task.dependencies == ["1"]
        assert task.status == "completed"
        assert task.result == "done"

    def test_task_model_id_required(self):
        with pytest.raises(ValidationError):
            TaskModel(description="test")

    def test_task_model_description_required(self):
        with pytest.raises(ValidationError):
            TaskModel(id="1")


class TestPlan:
    def test_plan_valid(self):
        plan = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test rationale",
        )
        assert len(plan.tasks) == 1
        assert plan.rationale == "test rationale"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="empty")
        assert plan.tasks == []

    def test_plan_rationale_required(self):
        with pytest.raises(ValidationError):
            Plan(tasks=[TaskModel(id="1", description="task")])


class TestTestResult:
    def test_test_result_passed(self):
        result = TestResult(passed=True, summary="all good", failures=[])
        assert result.passed is True
        assert result.summary == "all good"
        assert result.failures == []

    def test_test_result_failed(self):
        result = TestResult(passed=False, summary="failed", failures=["error 1", "error 2"])
        assert result.passed is False
        assert result.failures == ["error 1", "error 2"]

    def test_test_result_default_failures(self):
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []


class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"

    def test_provider_from_string(self):
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("openai") == Provider.OPENAI
        assert Provider("ollama") == Provider.OLLAMA


class TestSettings:
    def test_default_values(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.planner_model == "claude-3-5-haiku-20241022"
        assert settings.worker_model == "claude-3-5-sonnet-20241022"
        assert settings.tester_model == "claude-3-5-sonnet-20241022"
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.max_parallel_tasks == 5
        assert settings.max_cycles == 0
        assert settings.max_retries == 3
        assert settings.workspace == Path.cwd()
        assert settings.log_level == "INFO"
        assert settings.max_tokens == 4096
        assert settings.test_commands == []

    def test_env_prefix_furrow(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        monkeypatch.setenv("FURROW_MAX_TOKENS", "8192")

        settings = Settings()
        assert settings.max_cycles == 10
        assert settings.provider == Provider.OPENAI
        assert settings.max_tokens == 8192

    def test_env_override_max_parallel_tasks(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
        settings = Settings()
        assert settings.max_parallel_tasks == 3

    def test_env_override_workspace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FURROW_WORKSPACE", str(tmp_path))
        settings = Settings()
        assert settings.workspace == tmp_path

    def test_env_override_test_commands(self, monkeypatch):
        monkeypatch.setenv("FURROW_TEST_COMMANDS", "pytest -v, cargo test")
        settings = Settings()
        assert settings.test_commands == ["pytest -v", "cargo test"]

    def test_custom_settings(self):
        settings = Settings(
            provider=Provider.OLLAMA,
            model="llama3",
            max_cycles=5,
            max_parallel_tasks=2,
        )
        assert settings.provider == Provider.OLLAMA
        assert settings.model == "llama3"
        assert settings.max_cycles == 5
        assert settings.max_parallel_tasks == 2

    def test_settings_model_config_env_prefix(self):
        assert Settings.model_config["env_prefix"] == "FURROW_"

    def test_settings_env_file(self):
        assert Settings.model_config["env_file"] == ".env"


class TestSettingsIntegration:
    def test_roundtrip_dict(self):
        settings = Settings(max_cycles=7, provider=Provider.OPENAI)
        data = settings.model_dump()
        settings2 = Settings(**data)
        assert settings2.max_cycles == 7
        assert settings2.provider == Provider.OPENAI