import pytest
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Settings validation ---
class TestSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FURROW_MODEL"):
            monkeypatch.delenv(var, raising=False)
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.model == "claude-sonnet-4-20250514"
        assert s.planner_model == "claude-3-5-haiku-20241022"
        assert s.worker_model == "claude-3-5-sonnet-20241022"
        assert s.tester_model == "claude-3-5-sonnet-20241022"
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.log_level == "INFO"

    def test_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        monkeypatch.setenv("FURROW_MODEL", "gpt-4o")
        monkeypatch.setenv("FURROW_WORKER_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
        monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "2")
        monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.provider == Provider.OPENAI
        assert s.model == "gpt-4o"
        assert s.worker_model == "gpt-4o-mini"
        assert s.max_cycles == 10
        assert s.max_parallel_tasks == 2
        assert s.log_level == "DEBUG"


# --- Provider and TaskModel enums ---
class TestProviderEnum:
    def test_values(self) -> None:
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_iteration(self) -> None:
        names = {p.name for p in Provider}
        assert names == {"ANTHROPIC", "OPENAI", "OLLAMA"}


class TestTaskModel:
    def test_defaults(self) -> None:
        t = TaskModel(id="1", description="do thing")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_custom_values(self) -> None:
        t = TaskModel(
            id="1",
            description="do thing",
            files=["a.py"],
            dependencies=["0"],
            status="completed",
            result="done",
        )
        assert t.files == ["a.py"]
        assert t.dependencies == ["0"]
        assert t.status == "completed"
        assert t.result == "done"


# --- Plan and TestResult parsing/validation ---
class TestPlanValidation:
    def test_parse_with_one_task(self) -> None:
        data = {
            "tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}],
            "rationale": "ok",
        }
        p = Plan(**data)
        assert len(p.tasks) == 1
        assert p.rationale == "ok"

    def test_extra_fields_ignored_by_default(self) -> None:
        # Pydantic v2 default extra="ignore": unknown kwargs are silently discarded
        p = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok", extra=1)
        assert not hasattr(p, "extra")
        assert p.rationale == "ok"

    def test_tasks_default_empty(self) -> None:
        p = Plan(tasks=[], rationale="none")
        assert p.tasks == []


class TestTestResultValidation:
    def test_passed(self) -> None:
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True
        assert t.failures == []

    def test_failed(self) -> None:
        t = TestResult(passed=False, summary="bad", failures=["err1", "err2"])
        assert t.passed is False
        assert len(t.failures) == 2

    def test_default_failures_empty(self) -> None:
        t = TestResult(passed=True, summary="ok")
        assert t.failures == []
