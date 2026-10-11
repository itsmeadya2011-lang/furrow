import pytest
import os
from unittest.mock import patch
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_provider_enum_members(self):
        assert list(Provider) == [Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA]


class TestTaskModel:
    def test_task_model_defaults(self):
        task = TaskModel(id="1", description="test")
        assert task.id == "1"
        assert task.description == "test"
        assert task.files == []
        assert task.dependencies == []
        assert task.complexity == "medium"
        assert task.status == "pending"
        assert task.result is None

    def test_task_model_with_all_fields(self):
        task = TaskModel(
            id="1",
            description="test",
            files=["a.py", "b.py"],
            dependencies=["2"],
            complexity="high",
            status="completed",
            result="done"
        )
        assert task.files == ["a.py", "b.py"]
        assert task.dependencies == ["2"]
        assert task.complexity == "high"
        assert task.status == "completed"
        assert task.result == "done"

    def test_task_model_validation(self):
        with pytest.raises(ValueError):
            TaskModel(description="missing id")


class TestPlan:
    def test_plan_validation(self):
        plan = Plan(tasks=[TaskModel(id="1", description="test")], rationale="test")
        assert len(plan.tasks) == 1
        assert plan.rationale == "test"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="no tasks")
        assert plan.tasks == []

    def test_plan_multiple_tasks(self):
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ]
        plan = Plan(tasks=tasks, rationale="two tasks")
        assert len(plan.tasks) == 2


class TestTestResult:
    def test_test_result_passed(self):
        result = TestResult(passed=True, summary="all good", failures=[])
        assert result.passed is True
        assert result.failures == []

    def test_test_result_failed(self):
        result = TestResult(
            passed=False,
            summary="failed",
            failures=["test1 failed", "test2 failed"]
        )
        assert result.passed is False
        assert len(result.failures) == 2

    def test_test_result_default_failures(self):
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []


class TestSettings:
    def test_settings_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.provider == Provider.ANTHROPIC
            assert settings.model == "claude-sonnet-4-20250514"
            assert settings.planner_model == "claude-3-5-haiku-20241022"
            assert settings.worker_model == "claude-3-5-sonnet-20241022"
            assert settings.tester_model == "claude-3-5-sonnet-20241022"
            assert settings.max_parallel_tasks == 5
            assert settings.max_cycles == 0

    def test_settings_from_env_provider(self):
        with patch.dict(os.environ, {"FURROW_PROVIDER": "openai"}, clear=True):
            settings = Settings()
            assert settings.provider == Provider.OPENAI

    def test_settings_from_env_model(self):
        with patch.dict(os.environ, {"FURROW_MODEL": "gpt-4"}, clear=True):
            settings = Settings()
            assert settings.model == "gpt-4"

    def test_settings_from_env_api_keys(self):
        with patch.dict(os.environ, {
            "FURROW_ANTHROPIC_API_KEY": "anthropic-key",
            "FURROW_OPENAI_API_KEY": "openai-key",
        }, clear=True):
            settings = Settings()
            assert settings.anthropic_api_key == "anthropic-key"
            assert settings.openai_api_key == "openai-key"

    def test_settings_from_env_max_cycles(self):
        with patch.dict(os.environ, {"FURROW_MAX_CYCLES": "10"}, clear=True):
            settings = Settings()
            assert settings.max_cycles == 10

    def test_settings_from_env_max_parallel(self):
        with patch.dict(os.environ, {"FURROW_MAX_PARALLEL_TASKS": "3"}, clear=True):
            settings = Settings()
            assert settings.max_parallel_tasks == 3

    def test_settings_from_env_log_level(self):
        with patch.dict(os.environ, {"FURROW_LOG_LEVEL": "DEBUG"}, clear=True):
            settings = Settings()
            assert settings.log_level == "DEBUG"

    def test_settings_ollama_base_url(self):
        with patch.dict(os.environ, {"FURROW_OLLAMA_BASE_URL": "http://custom:11434"}, clear=True):
            settings = Settings()
            assert settings.ollama_base_url == "http://custom:11434"

    def test_settings_workspace(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.workspace.exists()

    def test_settings_explicit_override_env(self):
        with patch.dict(os.environ, {"FURROW_PROVIDER": "ollama"}, clear=True):
            settings = Settings(provider=Provider.OPENAI)
            assert settings.provider == Provider.OPENAI