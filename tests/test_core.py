"""Tests for core config models and settings."""
import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# TaskModel tests
# ---------------------------------------------------------------------------


def test_task_model_defaults():
    """TaskModel should have correct default values."""
    t = TaskModel(id="1", description="test task")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_full():
    """TaskModel should accept all fields."""
    t = TaskModel(
        id="t1",
        description="Implement auth",
        files=["src/auth.py", "tests/test_auth.py"],
        dependencies=["t0"],
        status="completed",
        result="Done",
    )
    assert t.id == "t1"
    assert t.files == ["src/auth.py", "tests/test_auth.py"]
    assert t.dependencies == ["t0"]
    assert t.status == "completed"
    assert t.result == "Done"


# ---------------------------------------------------------------------------
# TestResult tests
# ---------------------------------------------------------------------------


def test_test_result_with_failures():
    """TestResult should store failure list."""
    t = TestResult(passed=False, summary="2 failures", failures=["err1", "err2"])
    assert t.passed is False
    assert len(t.failures) == 2
    assert t.failures == ["err1", "err2"]


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


def test_settings_defaults():
    """Settings should have correct default values."""
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.max_tokens == 4096
    assert s.temperature == 0.7
    assert s.state_file == "furrow_state.json"
    assert s.log_level == "INFO"


def test_settings_custom_values():
    """Settings should accept custom values."""
    s = Settings(
        provider=Provider.OPENAI,
        model="gpt-4",
        max_cycles=3,
        max_parallel_tasks=10,
        max_tokens=2048,
        temperature=0.5,
    )
    assert s.provider == Provider.OPENAI
    assert s.model == "gpt-4"
    assert s.max_cycles == 3
    assert s.max_parallel_tasks == 10
    assert s.max_tokens == 2048
    assert s.temperature == 0.5


def test_settings_provider_enum_values():
    """Provider enum should have the correct values."""
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_settings_ollama_provider():
    """Settings should accept OLLAMA as provider."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://localhost:11434"
