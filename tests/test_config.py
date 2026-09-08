import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.model == "claude-sonnet-4-20250514"
        assert s.planner_model == "claude-3-5-haiku-20241022"
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0

    def test_custom_env_vars(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FURROW_PROVIDER", "ollama")
        monkeypatch.setenv("FURROW_MODEL", "llama3")
        monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
        monkeypatch.setenv("FURROW_WORKSPACE", str(tmp_path))

        s = Settings()
        assert s.provider == Provider.OLLAMA
        assert s.model == "llama3"
        assert s.max_cycles == 10
        assert s.workspace == tmp_path

    def test_provider_string_override(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        s = Settings()
        assert s.provider == Provider.OPENAI

    def test_invalid_provider(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "invalid")
        with pytest.raises(ValidationError):
            Settings()

    def test_state_file_default(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        s = Settings()
        assert s.state_file == Path("furrow_state.json")

    def test_api_key_env_fallback(self, monkeypatch):
        monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        s = Settings()
        assert s.anthropic_api_key is None


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_with_values(self):
        t = TaskModel(
            id="t1",
            description="write tests",
            files=["tests.py"],
            dependencies=["t0"],
            status="completed",
            result="done",
        )
        assert t.files == ["tests.py"]
        assert t.dependencies == ["t0"]
        assert t.status == "completed"
        assert t.result == "done"


class TestPlan:
    def test_parse_plan(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"
        assert p.rationale == "ok"

    def test_empty_tasks(self):
        p = Plan(tasks=[], rationale="nothing to do")
        assert p.tasks == []


class TestTestResult:
    def test_passed_result(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True
        assert t.summary == "ok"

    def test_failed_result(self):
        t = TestResult(passed=False, summary="bad", failures=["error1"])
        assert t.passed is False
        assert t.failures == ["error1"]

    def test_default_failures(self):
        t = TestResult(passed=True, summary="ok")
        assert t.failures == []
