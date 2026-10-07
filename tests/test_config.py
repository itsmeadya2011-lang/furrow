import pytest
from pathlib import Path

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)


class TestTaskModel:
    def test_task_creation_with_required_fields(self):
        task = TaskModel(id="1", description="do thing")
        assert task.id == "1"
        assert task.description == "do thing"

    def test_task_defaults_status_pending(self):
        task = TaskModel(id="1", description="do thing")
        assert task.status == "pending"

    def test_task_defaults_result_none(self):
        task = TaskModel(id="1", description="do thing")
        assert task.result is None

    def test_task_defaults_empty_files(self):
        task = TaskModel(id="1", description="do thing")
        assert task.files == []

    def test_task_defaults_empty_dependencies(self):
        task = TaskModel(id="1", description="do thing")
        assert task.dependencies == []

    def test_task_with_files_and_dependencies(self):
        task = TaskModel(
            id="2",
            description="implement feature",
            files=["src/main.py", "tests/test_main.py"],
            dependencies=["1"],
        )
        assert task.files == ["src/main.py", "tests/test_main.py"]
        assert task.dependencies == ["1"]

    def test_task_with_custom_status_and_result(self):
        task = TaskModel(
            id="3",
            description="fix bug",
            status="completed",
            result="Fixed the issue",
        )
        assert task.status == "completed"
        assert task.result == "Fixed the issue"

    def test_task_with_failed_status(self):
        task = TaskModel(id="4", description="failing task", status="failed")
        assert task.status == "failed"


class TestPlan:
    def test_plan_creation(self):
        plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="because")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"
        assert plan.rationale == "because"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="nothing to do")
        assert plan.tasks == []

    def test_plan_multiple_tasks(self):
        tasks = [TaskModel(id=str(i), description=f"task {i}") for i in range(3)]
        plan = Plan(tasks=tasks, rationale="multiple")
        assert len(plan.tasks) == 3


class TestTestResult:
    def test_test_result_passed(self):
        result = TestResult(passed=True, summary="All tests passed")
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    def test_test_result_failed(self):
        result = TestResult(
            passed=False,
            summary="Tests failed",
            failures=["test_foo.py::test_bar - AssertionError"],
        )
        assert result.passed is False
        assert len(result.failures) == 1
        assert "AssertionError" in result.failures[0]

    def test_test_result_failed_no_failures_list(self):
        result = TestResult(passed=False, summary="error")
        assert result.passed is False
        assert result.failures == []

    def test_test_result_with_multiple_failures(self):
        failures = ["fail1", "fail2", "fail3"]
        result = TestResult(passed=False, summary="multiple failures", failures=failures)
        assert len(result.failures) == 3


class TestSettings:
    def test_settings_defaults_provider(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC

    def test_settings_defaults_model(self):
        settings = Settings()
        assert settings.model == "claude-sonnet-4-20250514"

    def test_settings_defaults_planner_model(self):
        settings = Settings()
        assert settings.planner_model == "claude-3-5-haiku-20241022"

    def test_settings_defaults_worker_model(self):
        settings = Settings()
        assert settings.worker_model == "claude-3-5-sonnet-20241022"

    def test_settings_defaults_tester_model(self):
        settings = Settings()
        assert settings.tester_model == "claude-3-5-sonnet-20241022"

    def test_settings_defaults_max_parallel_tasks(self):
        settings = Settings()
        assert settings.max_parallel_tasks == 5

    def test_settings_defaults_max_cycles(self):
        settings = Settings()
        assert settings.max_cycles == 0

    def test_settings_defaults_ollama_base_url(self):
        settings = Settings()
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_settings_defaults_log_level(self):
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_settings_api_keys_default_none(self):
        settings = Settings()
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None

    def test_settings_workspace_default_cwd(self):
        settings = Settings()
        assert settings.workspace == Path.cwd()

    def test_settings_provider_can_be_overridden(self):
        settings = Settings(provider=Provider.OPENAI)
        assert settings.provider == Provider.OPENAI

    def test_settings_max_parallel_tasks_override(self):
        settings = Settings(max_parallel_tasks=10)
        assert settings.max_parallel_tasks == 10

    def test_settings_max_cycles_override(self):
        settings = Settings(max_cycles=3)
        assert settings.max_cycles == 3

    def test_settings_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        settings = Settings()
        assert settings.provider == Provider.OPENAI

    def test_settings_provider_from_env_ollama(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "ollama")
        settings = Settings()
        assert settings.provider == Provider.OLLAMA

    def test_settings_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "test-key-123")
        settings = Settings()
        assert settings.anthropic_api_key == "test-key-123"