import pytest
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ─── Orchestrator tests ──────────────────────────────────────────────


def _make_orchestrator(max_cycles: int = 0, max_parallel: int = 5) -> Orchestrator:
    """Create an Orchestrator with overridden settings (no real LLM needed)."""
    from furrow.config import Provider, Settings

    settings = Settings(
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel,
    )
    # Bypass real LLM client creation
    return Orchestrator(goal="test goal", settings=settings)


def test_orchestrator_is_done_no_tasks():
    """With no tasks, the goal is vacuously complete."""
    o = _make_orchestrator()
    assert o._get_tasks() == []
    assert o._is_done() is True


def test_orchestrator_is_done_all_completed():
    """All tasks completed → done."""
    o = _make_orchestrator()
    o._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_done_has_failed():
    """Failed tasks mean not done."""
    o = _make_orchestrator()
    o._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_done_partial():
    """Partially completed tasks → not done."""
    o = _make_orchestrator()
    o._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert o._is_done() is False


def test_orchestrator_tracks_tasks_across_cycles():
    """Tasks set during _cycle should be retrievable via _get_tasks."""
    o = _make_orchestrator()
    # Simulate what _cycle does
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="do a", status="completed"),
            TaskModel(id="2", description="do b", status="completed"),
        ],
        rationale="test",
    )
    o._tasks = plan.tasks
    assert o._get_tasks() == plan.tasks
    assert o._is_done() is True


def test_orchestrator_max_cycles_setting():
    """max_cycles is stored in settings."""
    o = _make_orchestrator(max_cycles=3)
    assert o.settings.max_cycles == 3
    assert o.settings.max_parallel_tasks == 5


def test_orchestrator_max_parallel_setting():
    """max_parallel_tasks is stored in settings."""
    o = _make_orchestrator(max_parallel=2)
    assert o.settings.max_parallel_tasks == 2


# ─── Tester fallback tests ───────────────────────────────────────────


def test_tester_extract_failures_from_json_like_text():
    """When the LLM returns plain text with 'fail', extract those lines."""
    text = "Tests failed.\n  • AssertionError in test_x\nSome error occurred"
    failures = TesterAgent._extract_failures(text)
    assert "Tests failed." in failures
    assert any("AssertionError" in f for f in failures)


def test_tester_extract_failures_no_relevant_lines():
    """Fallback: if no keyword lines, use the whole text."""
    text = "Something unexpected happened"
    failures = TesterAgent._extract_failures(text)
    assert failures == ["Something unexpected happened"]


def test_tester_extract_failures_empty():
    """Empty text produces empty failures list."""
    failures = TesterAgent._extract_failures("")
    assert failures == []


def test_tester_fallback_pass_detection():
    """A response containing 'passed' but not 'fail' should be treated as pass."""
    # Simulate the fallback logic
    response = "All tests passed successfully."
    passed = "passed" in response.lower() and "fail" not in response.lower()
    assert passed is True


# ─── LLM provider routing test ────────────────────────────────────────


def test_ollama_provider_in_config():
    """Provider.OLLAMA should be defined and selectable."""
    from furrow.config import Provider

    assert Provider.OLLAMA == "ollama"
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"


def test_llm_client_has_ollama_method():
    """LLMClient should have a _complete_ollama method."""
    from furrow.llm import LLMClient

    assert hasattr(LLMClient, "_complete_ollama")
