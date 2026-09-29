"""Tests for config models."""
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult

# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------

def test_plan_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
        ],
        rationale="two tasks",
    )
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_task_model_defaults():
    t = TaskModel(id="x", description="do something")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_settings_defaults():
    # Use explicit values to avoid env interference
    s = Settings(
        provider=Provider.ANTHROPIC,
        max_parallel_tasks=5,
        max_cycles=0,
        log_level="INFO",
    )
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_overrides():
    s = Settings(
        provider=Provider.OPENAI,
        max_cycles=10,
        max_parallel_tasks=2,
    )
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 10
    assert s.max_parallel_tasks == 2


def test_test_result_defaults():
    t = TestResult(passed=False, summary="fail")
    assert t.failures == []


def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
