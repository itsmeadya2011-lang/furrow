"""Tests for furrow config types."""
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


class TestProvider:
    def test_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_default_is_anthropic(self):
        assert Provider.ANTHROPIC is not None


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="test task")
        assert t.status == "pending"
        assert t.files == []
        assert t.dependencies == []
        assert t.result is None

    def test_with_files(self):
        t = TaskModel(id="2", description="task with files", files=["a.py", "b.py"])
        assert t.files == ["a.py", "b.py"]


class TestPlan:
    def test_valid_plan(self):
        tasks = [TaskModel(id="1", description="do thing")]
        p = Plan(tasks=tasks, rationale="ok")
        assert len(p.tasks) == 1
        assert p.rationale == "ok"

    def test_empty_plan(self):
        p = Plan(tasks=[], rationale="nothing to do")
        assert p.tasks == []


class TestTestResult:
    def test_passed(self):
        t = TestResult(passed=True, summary="All good")
        assert t.passed is True
        assert t.failures == []

    def test_failed(self):
        t = TestResult(passed=False, summary="Broken", failures=["test_a failed"])
        assert t.passed is False
        assert len(t.failures) == 1


class TestSettings:
    def test_defaults(self, monkeypatch):
        # Ensure no env vars interfere
        for key in ["FURROW_PROVIDER", "FURROW_MODEL", "FURROW_MAX_CYCLES"]:
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_cycles == 0
        assert s.max_parallel_tasks == 5
        assert s.ollama_base_url == "http://localhost:11434"

    def test_max_cycles_from_env(self, monkeypatch):
        monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
        s = Settings()
        assert s.max_cycles == 10
