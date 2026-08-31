import os
from pathlib import Path
from unittest.mock import patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestSettings:
    """Tests for the Settings class."""

    def test_settings_defaults(self):
        """Test Settings defaults."""
        # Clear any FURROW_ env vars that might interfere
        with patch.dict(os.environ, {}, clear=False):
            # Remove FURROW_ prefixed vars
            env_backup = {}
            for key in list(os.environ.keys()):
                if key.startswith("FURROW_"):
                    env_backup[key] = os.environ.pop(key)
            try:
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
                assert settings.log_level == "INFO"
            finally:
                os.environ.update(env_backup)

    def test_settings_reads_from_env_vars(self):
        """Test Settings reads from env vars."""
        env_vars = {
            "FURROW_PROVIDER": "openai",
            "FURROW_MODEL": "gpt-4",
            "FURROW_PLANNER_MODEL": "gpt-3.5-turbo",
            "FURROW_WORKER_MODEL": "gpt-4-turbo",
            "FURROW_TESTER_MODEL": "gpt-4",
            "FURROW_ANTHROPIC_API_KEY": "test-anthropic-key",
            "FURROW_OPENAI_API_KEY": "test-openai-key",
            "FURROW_OLLAMA_BASE_URL": "http://custom:8080",
            "FURROW_MAX_PARALLEL_TASKS": "10",
            "FURROW_MAX_CYCLES": "5",
            "FURROW_LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.provider == Provider.OPENAI
            assert settings.model == "gpt-4"
            assert settings.planner_model == "gpt-3.5-turbo"
            assert settings.worker_model == "gpt-4-turbo"
            assert settings.tester_model == "gpt-4"
            assert settings.anthropic_api_key == "test-anthropic-key"
            assert settings.openai_api_key == "test-openai-key"
            assert settings.ollama_base_url == "http://custom:8080"
            assert settings.max_parallel_tasks == 10
            assert settings.max_cycles == 5
            assert settings.log_level == "DEBUG"

    def test_settings_provider_enum(self):
        """Test Settings provider enum conversion."""
        with patch.dict(os.environ, {"FURROW_PROVIDER": "ollama"}):
            settings = Settings()
            assert settings.provider == Provider.OLLAMA

    def test_settings_workspace_default(self):
        """Test Settings workspace defaults to cwd."""
        with patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ.keys()):
                if key.startswith("FURROW_"):
                    os.environ.pop(key)
            settings = Settings()
            assert settings.workspace == Path.cwd()


class TestTaskModel:
    """Tests for the TaskModel class."""

    def test_task_status_defaults_to_pending(self):
        """Test TaskModel status defaults to pending."""
        task = TaskModel(id="1", description="test task")
        assert task.status == "pending"

    def test_task_files_defaults_to_empty_list(self):
        """Test TaskModel files defaults to empty list."""
        task = TaskModel(id="1", description="test task")
        assert task.files == []

    def test_task_dependencies_defaults_to_empty_list(self):
        """Test TaskModel dependencies defaults to empty list."""
        task = TaskModel(id="1", description="test task")
        assert task.dependencies == []

    def test_task_result_defaults_to_none(self):
        """Test TaskModel result defaults to None."""
        task = TaskModel(id="1", description="test task")
        assert task.result is None

    def test_task_with_all_fields(self):
        """Test TaskModel with all fields specified."""
        task = TaskModel(
            id="1",
            description="test task",
            files=["src/main.py"],
            dependencies=["0"],
            status="completed",
            result="done",
        )
        assert task.id == "1"
        assert task.description == "test task"
        assert task.files == ["src/main.py"]
        assert task.dependencies == ["0"]
        assert task.status == "completed"
        assert task.result == "done"


class TestPlan:
    """Tests for the Plan class."""

    def test_plan_validation(self):
        """Test Plan validation."""
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1"),
                TaskModel(id="2", description="task 2"),
            ],
            rationale="Test plan",
        )
        assert len(plan.tasks) == 2
        assert plan.rationale == "Test plan"

    def test_plan_empty_tasks_valid(self):
        """Test Plan with empty tasks is valid."""
        plan = Plan(tasks=[], rationale="No tasks needed")
        assert plan.tasks == []

    def test_plan_requires_rationale(self):
        """Test Plan requires rationale field."""
        with pytest.raises(Exception):  # ValidationError
            Plan(tasks=[TaskModel(id="1", description="task")])

    def test_plan_requires_tasks(self):
        """Test Plan requires tasks field."""
        with pytest.raises(Exception):  # ValidationError
            Plan(rationale="Missing tasks")

    def test_plan_task_validation(self):
        """Test Plan validates task fields."""
        # Task requires id and description
        with pytest.raises(Exception):  # ValidationError
            Plan(tasks=[TaskModel(description="missing id")], rationale="test")


class TestTestResult:
    """Tests for the TestResult class."""

    def test_test_result_creation(self):
        """Test TestResult creation."""
        result = TestResult(passed=True, summary="All tests passed", failures=[])
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    def test_test_result_with_failures(self):
        """Test TestResult with failures."""
        result = TestResult(
            passed=False,
            summary="Some tests failed",
            failures=["test_auth failed", "test_db failed"],
        )
        assert result.passed is False
        assert len(result.failures) == 2

    def test_test_result_failures_default(self):
        """Test TestResult failures defaults to empty list."""
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []
