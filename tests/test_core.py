"""Comprehensive tests for Furrow framework."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestSettings:
    def test_default_provider(self):
        settings = Settings()
        assert settings.provider == Provider.ANTHROPIC

    def test_default_max_parallel_tasks(self):
        settings = Settings()
        assert settings.max_parallel_tasks == 5

    def test_default_max_cycles(self):
        settings = Settings()
        assert settings.max_cycles == 0  # 0 = infinite

    def test_model_defaults(self):
        settings = Settings()
        assert settings.planner_model == "claude-3-5-haiku-20241022"
        assert settings.worker_model == "claude-3-5-sonnet-20241022"
        assert settings.tester_model == "claude-3-5-sonnet-20241022"

    def test_workspace_default(self):
        settings = Settings()
        assert isinstance(settings.workspace, Path)

    @patch.dict(os.environ, {"FURROW_PROVIDER": "openai", "FURROW_MAX_CYCLES": "10"})
    def test_env_prefix(self):
        """Settings should read FURROW_ prefixed env vars."""
        settings = Settings()
        assert settings.provider == Provider.OPENAI
        assert settings.max_cycles == 10

    @patch.dict(
        os.environ, {"FURROW_PROVIDER": "ollama", "FURROW_OLLAMA_BASE_URL": "http://host:1234"}
    )
    def test_ollama_env_prefix(self):
        settings = Settings()
        assert settings.provider == Provider.OLLAMA
        assert settings.ollama_base_url == "http://host:1234"

    def test_bare_kwargs(self):
        """Settings should accept bare field names as kwargs."""
        settings = Settings(provider=Provider.OPENAI, openai_api_key="mykey", max_cycles=50)
        assert settings.provider == Provider.OPENAI
        assert settings.openai_api_key == "mykey"
        assert settings.max_cycles == 50


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.id == "1"
        assert t.description == "do thing"
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_with_files(self):
        t = TaskModel(id="2", description="do other", files=["a.py", "b.py"])
        assert t.files == ["a.py", "b.py"]

    def test_status_transition(self):
        t = TaskModel(id="1", description="do thing")
        t.status = "completed"
        assert t.status == "completed"

    def test_validation_id_required(self):
        with pytest.raises(ValidationError):
            TaskModel(description="no id")

    def test_validation_description_required(self):
        with pytest.raises(ValidationError):
            TaskModel(id="1")


class TestPlan:
    def test_valid_plan(self):
        plan = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test rationale",
        )
        assert len(plan.tasks) == 1
        assert plan.rationale == "test rationale"

    def test_empty_tasks(self):
        plan = Plan(tasks=[], rationale="nothing to do")
        assert plan.tasks == []

    def test_from_dict(self):
        data = {
            "tasks": [
                {"id": "1", "description": "build auth", "files": ["auth.py"], "dependencies": []},
                {
                    "id": "2",
                    "description": "add tests",
                    "files": ["test_auth.py"],
                    "dependencies": ["1"],
                },
            ],
            "rationale": "build auth end-to-end",
        }
        plan = Plan(**data)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[1].dependencies == ["1"]


class TestTestResult:
    def test_passing_result(self):
        t = TestResult(passed=True, summary="all good", failures=[])
        assert t.passed is True

    def test_failing_result(self):
        t = TestResult(passed=False, summary="2 failures", failures=["fail 1", "fail 2"])
        assert t.passed is False
        assert len(t.failures) == 2

    def test_default_failures(self):
        t = TestResult(passed=True, summary="ok")
        assert t.failures == []

    def test_from_dict(self):
        data = {"passed": False, "summary": "broken", "failures": ["x"]}
        t = TestResult(**data)
        assert t.passed is False


class TestProvider:
    def test_provider_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"

    def test_provider_from_string(self):
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("openai") == Provider.OPENAI
        assert Provider("ollama") == Provider.OLLAMA
