from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import (
    Completion,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    ToolCall,
)


class TestProviderEnum:
    def test_provider_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_provider_from_string(self):
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("openai") == Provider.OPENAI
        assert Provider("ollama") == Provider.OLLAMA


class TestTaskModel:
    def test_task_model_creation_with_defaults(self):
        task = TaskModel(id="1", description="do thing")
        assert task.id == "1"
        assert task.description == "do thing"
        assert task.files == []
        assert task.dependencies == []
        assert task.status == "pending"
        assert task.result is None

    def test_task_model_with_dependencies(self):
        task = TaskModel(
            id="2",
            description="depends on 1",
            files=["src/main.py"],
            dependencies=["1"],
            status="in_progress",
            result="done",
        )
        assert task.files == ["src/main.py"]
        assert task.dependencies == ["1"]
        assert task.status == "in_progress"
        assert task.result == "done"

    def test_task_model_validation_error_on_missing_id(self):
        with pytest.raises(ValidationError):
            TaskModel(description="no id")

    def test_task_model_validation_error_on_missing_description(self):
        with pytest.raises(ValidationError):
            TaskModel(id="1")


class TestPlan:
    def test_plan_model_validation(self):
        tasks = [TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")]
        plan = Plan(tasks=tasks, rationale="test rationale")
        assert len(plan.tasks) == 2
        assert plan.rationale == "test rationale"

    def test_plan_empty_tasks(self):
        plan = Plan(tasks=[], rationale="empty plan")
        assert plan.tasks == []

    def test_plan_validation_error_missing_rationale(self):
        with pytest.raises(ValidationError):
            Plan(tasks=[TaskModel(id="1", description="task 1")])


class TestTestResult:
    def test_test_result_model(self):
        result = TestResult(passed=True, summary="all good", failures=[])
        assert result.passed is True
        assert result.summary == "all good"
        assert result.failures == []

    def test_test_result_with_failures(self):
        result = TestResult(passed=False, summary="failed", failures=["test 1 failed", "test 2 failed"])
        assert result.passed is False
        assert result.failures == ["test 1 failed", "test 2 failed"]

    def test_test_result_default_failures(self):
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []


class TestToolCall:
    def test_tool_call_creation(self):
        tool = ToolCall(name="read_file", arguments={"path": "test.py"}, id="call_123")
        assert tool.name == "read_file"
        assert tool.arguments == {"path": "test.py"}
        assert tool.id == "call_123"


class TestCompletion:
    def test_completion_defaults(self):
        completion = Completion()
        assert completion.text == ""
        assert completion.tool_calls == []
        assert completion.finish_reason is None
        assert completion.usage is None

    def test_completion_with_tool_calls(self):
        tool = ToolCall(name="write_file", arguments={"path": "test.py", "content": "hi"}, id="call_1")
        completion = Completion(text="done", tool_calls=[tool], finish_reason="tool_use")
        assert completion.text == "done"
        assert len(completion.tool_calls) == 1
        assert completion.finish_reason == "tool_use"


class TestSettings:
    def test_settings_defaults(self):
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
        assert isinstance(settings.workspace, Path)
        assert settings.log_level == "INFO"
        assert settings.test_command is None
        assert settings.test_timeout == 120
        assert settings.max_tokens == 4096
        assert settings.max_retries == 3
        assert settings.timeout == 120.0

    def test_settings_custom_env_vars(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        monkeypatch.setenv("FURROW_MODEL", "gpt-4")
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
        monkeypatch.setenv("FURROW_MAX_CYCLES", "5")
        monkeypatch.setenv("FURROW_WORKSPACE", "/custom/workspace")
        monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("FURROW_TEST_COMMAND", "pytest -v")
        monkeypatch.setenv("FURROW_TEST_TIMEOUT", "300")
        monkeypatch.setenv("FURROW_MAX_TOKENS", "8192")
        monkeypatch.setenv("FURROW_MAX_RETRIES", "5")
        monkeypatch.setenv("FURROW_TIMEOUT", "60.0")

        settings = Settings()
        assert settings.provider == Provider.OPENAI
        assert settings.model == "gpt-4"
        assert settings.max_parallel_tasks == 10
        assert settings.max_cycles == 5
        assert settings.workspace == Path("/custom/workspace")
        assert settings.log_level == "DEBUG"
        assert settings.test_command == "pytest -v"
        assert settings.test_timeout == 300
        assert settings.max_tokens == 8192
        assert settings.max_retries == 5
        assert settings.timeout == 60.0

    def test_settings_anthropic_api_key_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "test-key")
        settings = Settings()
        assert settings.anthropic_api_key == "test-key"

    def test_settings_openai_api_key_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_OPENAI_API_KEY", "test-key")
        settings = Settings()
        assert settings.openai_api_key == "test-key"