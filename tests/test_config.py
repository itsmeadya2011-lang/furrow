"""Tests for the Furrow config module."""

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_provider_enum():
    """Provider enum should have the expected values."""
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_empty():
    p = Plan(tasks=[], rationale="nothing to do")
    assert len(p.tasks) == 0
    assert p.rationale == "nothing to do"


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_files():
    t = TaskModel(id="2", description="fix", files=["src/app.py", "tests/test_app.py"], dependencies=["1"])
    assert t.files == ["src/app.py", "tests/test_app.py"]
    assert t.dependencies == ["1"]


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failures", failures=["test_x failed"])
    assert t.passed is False
    assert "test_x failed" in t.failures


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    s = Settings()
    assert s.provider == Provider.OPENAI
