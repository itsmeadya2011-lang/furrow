"""Tests for config models and settings."""
from __future__ import annotations

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.id == "1"
        assert t.description == "do thing"
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None


class TestPlan:
    def test_from_dict(self):
        p = Plan(
            tasks=[TaskModel(id="1", description="task1", files=[], dependencies=[])],
            rationale="test",
        )
        assert len(p.tasks) == 1
        assert p.tasks[0].description == "task1"
        assert p.rationale == "test"


class TestTestResult:
    def test_passed_defaults(self):
        t = TestResult(passed=True, summary="ok")
        assert t.passed is True
        assert t.failures == []


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.ollama_base_url == "http://localhost:11434"

    def test_provider_ollama(self):
        s = Settings(provider=Provider.OLLAMA)
        assert s.provider == Provider.OLLAMA

    def test_workspace_path(self, tmp_path):
        s = Settings(workspace=tmp_path)
        assert s.workspace == tmp_path


class TestProvider:
    def test_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"
