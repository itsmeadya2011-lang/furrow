from pathlib import Path

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)


class TestTaskModel:
    def test_default_values(self):
        task = TaskModel(id="1", description="test task")
        assert task.status == "pending"
        assert task.files == []
        assert task.dependencies == []
        assert task.result is None

    def test_custom_values(self):
        task = TaskModel(
            id="2",
            description="custom task",
            files=["file1.py", "file2.py"],
            dependencies=["1"],
            status="completed",
            result="success",
        )
        assert task.status == "completed"
        assert task.files == ["file1.py", "file2.py"]
        assert task.dependencies == ["1"]
        assert task.result == "success"

    def test_optional_fields_can_be_none(self):
        task = TaskModel(id="3", description="task", result=None)
        assert task.result is None


class TestPlan:
    def test_plan_with_tasks_and_rationale(self):
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ]
        plan = Plan(tasks=tasks, rationale="test rationale")
        assert len(plan.tasks) == 2
        assert plan.rationale == "test rationale"
        assert plan.tasks[0].id == "1"
        assert plan.tasks[1].id == "2"

    def test_empty_tasks_list(self):
        plan = Plan(tasks=[], rationale="no tasks")
        assert plan.tasks == []


class TestTestResult:
    def test_passed_with_empty_failures(self):
        result = TestResult(passed=True, summary="all good", failures=[])
        assert result.passed is True
        assert result.failures == []

    def test_failed_with_failures(self):
        result = TestResult(
            passed=False,
            summary="tests failed",
            failures=["test_foo failed", "test_bar failed"],
        )
        assert result.passed is False
        assert len(result.failures) == 2

    def test_default_failures_list(self):
        result = TestResult(passed=True, summary="ok")
        assert result.failures == []


class TestSettings:
    def test_default_provider(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC

    def test_default_models(self):
        settings = Settings()
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.planner_model == "claude-3-5-haiku-20241022"
        assert settings.worker_model == "claude-3-5-sonnet-20241022"
        assert settings.tester_model == "claude-3-5-sonnet-20241022"

    def test_default_max_parallel_tasks(self):
        settings = Settings()
        assert settings.max_parallel_tasks == 5

    def test_default_max_cycles(self):
        settings = Settings()
        assert settings.max_cycles == 0

    def test_default_workspace(self):
        settings = Settings()
        assert settings.workspace == Path.cwd()

    def test_custom_settings(self):
        settings = Settings(
            provider=Provider.OPENAI,
            model="gpt-4o",
            max_parallel_tasks=10,
            max_cycles=5,
            workspace=Path("/custom/path"),
        )
        assert settings.provider == Provider.OPENAI
        assert settings.model == "gpt-4o"
        assert settings.max_parallel_tasks == 10
        assert settings.max_cycles == 5
        assert settings.workspace == Path("/custom/path")

    def test_ollama_base_url_default(self):
        settings = Settings()
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_log_level_default(self):
        settings = Settings()
        assert settings.log_level == "INFO"