"""Tests for configuration models and settings."""
import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"

    def test_provider_is_str_enum(self):
        assert Provider.ANTHROPIC == "anthropic"


class TestTaskModel:
    def test_defaults(self):
        task = TaskModel(id="1", description="do thing")
        assert task.status == "pending"
        assert task.files == []
        assert task.dependencies == []
        assert task.result is None

    def test_with_files(self):
        task = TaskModel(
            id="2",
            description="implement feature",
            files=["src/app.py", "tests/test_app.py"],
            dependencies=["1"],
        )
        assert task.files == ["src/app.py", "tests/test_app.py"]
        assert task.dependencies == ["1"]


class TestPlan:
    def test_valid_plan(self):
        plan = Plan(
            tasks=[TaskModel(id="1", description="task one")],
            rationale="Test plan rationale",
        )
        assert len(plan.tasks) == 1
        assert plan.rationale == "Test plan rationale"

    def test_empty_plan(self):
        plan = Plan(tasks=[], rationale="No tasks needed")
        assert plan.tasks == []
        assert plan.rationale == "No tasks needed"


class TestTestResult:
    def test_passed_result(self):
        result = TestResult(passed=True, summary="All tests passed")
        assert result.passed is True
        assert result.failures == []

    def test_failed_result(self):
        result = TestResult(passed=False, summary="2 tests failed", failures=["test_a", "test_b"])
        assert result.passed is False
        assert len(result.failures) == 2

    def test_from_dict(self):
        data = {"passed": True, "summary": "ok", "failures": []}
        result = TestResult(**data)
        assert result.passed is True


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.ollama_base_url == "http://localhost:11434"
        assert s.ollama_model == "llama3.1"

    def test_provider_override_via_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        s = Settings()
        assert s.provider == Provider.OPENAI

    def test_max_cycles_override_via_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_CYCLES", "3")
        s = Settings()
        assert s.max_cycles == 3

    def test_host_port_defaults(self):
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8000
