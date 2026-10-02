import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="implement x")
    assert t.id == "1"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_dependencies():
    t = TaskModel(id="2", description="b", dependencies=["1"])
    assert t.dependencies == ["1"]


def test_plan_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b", dependencies=["1"]),
        ],
        rationale="multi",
    )
    assert len(p.tasks) == 2


def test_test_result_failures():
    t = TestResult(passed=False, summary="2 failing", failures=["test_a failed", "test_b failed"])
    assert not t.passed
    assert len(t.failures) == 2


def test_plan_empty():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []


# --- Config / Settings ---

from furrow.config import Provider, Settings


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_tool_rounds == 15
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.log_level == "INFO"


def test_settings_custom_values():
    s = Settings(max_cycles=3, max_parallel_tasks=2, log_level="DEBUG")
    assert s.max_cycles == 3
    assert s.max_parallel_tasks == 2
    assert s.log_level == "DEBUG"


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
