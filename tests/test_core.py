from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Orchestrator tests ---


def _make_orchestrator(max_cycles=0, max_parallel=5):
    """Build an Orchestrator with a mocked LLMClient."""
    mock_client = MagicMock()
    mock_client.settings.max_cycles = max_cycles
    mock_client.settings.max_parallel_tasks = max_parallel
    mock_client.settings.model = "test-model"
    mock_client.settings.planner_model = "planner-model"
    mock_client.settings.worker_model = "worker-model"
    mock_client.settings.tester_model = "tester-model"
    mock_client.settings.provider = MagicMock()
    # complete() returns a plain async stub
    mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
    return Orchestrator(goal="test goal", client=mock_client)


def test_orchestrator_starts_with_empty_tasks():
    orch = _make_orchestrator()
    assert orch._tasks == []
    assert orch.cycles == 0


def test_orchestrator_is_done_empty():
    """With no tasks, the orchestrator is NOT done (avoids premature halt)."""
    orch = _make_orchestrator()
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    orch = _make_orchestrator()
    orch._tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ]
    for t in orch._tasks:
        t.status = "completed"
    assert orch._is_done() is True


def test_orchestrator_is_done_has_failed():
    orch = _make_orchestrator()
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_is_done_partial():
    orch = _make_orchestrator()
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    """When max_cycles=1, the orchestrator runs at most 1 cycle and stops."""
    orch = _make_orchestrator(max_cycles=1)

    async def mock_plan(goal):
        return Plan(
            tasks=[TaskModel(id="1", description="task")],
            rationale="test",
        )

    orch.planner.plan = mock_plan

    call_tracker = MagicMock()

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )

        original_cycle = orch._cycle

        async def counting_cycle():
            call_tracker()
            await original_cycle()

        orch._cycle = counting_cycle
        await orch.run()

    assert call_tracker.call_count == 1
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_full_cycle_with_completed_tasks():
    """A full cycle should store tasks and mark them completed."""
    orch = _make_orchestrator()

    async def mock_plan(goal):
        return Plan(
            tasks=[TaskModel(id="1", description="task one")],
            rationale="test",
        )

    orch.planner.plan = mock_plan

    with patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="implemented")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all tests pass", failures=[])
        )

        await orch._cycle()

    # Tasks should be stored on the orchestrator
    assert len(orch._tasks) == 1
    assert orch._tasks[0].status == "completed"
    assert orch._tasks[0].result == "implemented"
    # With all tasks completed and tests passing, we should be done
    assert orch._is_done() is True


# --- LLMClient OLLAMA tests ---


def test_llm_client_ollama_provider():
    """LLMClient should have an ollama property that creates an httpx client."""
    from furrow.config import Provider

    s = MagicMock()
    s.provider = Provider.OLLAMA
    s.ollama_base_url = "http://localhost:11434"
    s.anthropic_api_key = None
    s.openai_api_key = None

    client = LLMClient(settings=s)
    # Accessing the property should create the httpx client
    ollama_client = client.ollama
    assert ollama_client is not None
    assert str(ollama_client.base_url).rstrip("/") == "http://localhost:11434"


def test_llm_client_ollama_supported_in_complete():
    """The complete() method should route OLLAMA to _complete_ollama."""
    from furrow.config import Provider

    s = MagicMock()
    s.provider = Provider.OLLAMA
    s.ollama_base_url = "http://localhost:11434"
    s.anthropic_api_key = None
    s.openai_api_key = None
    s.model = "llama3"

    client = LLMClient(settings=s)
    assert client._complete_ollama is not None


# --- Tester fallback tests ---


def test_tester_fallback_pass_detection():
    """When JSON parsing fails, the tester should try to infer pass/fail from text."""
    raw_result = TestResult(passed=True, summary="Tests passed", failures=[])
    # Simulate non-JSON response that contains "passed"
    response = "All tests passed successfully"
    passed = "passed" in response.lower() and "fail" not in response.lower()
    assert passed is True


def test_tester_fallback_fail_detection():
    """Non-JSON response with 'fail' should be treated as failure."""
    response = "Tests failed: 3 errors"
    passed = "passed" in response.lower() and "fail" not in response.lower()
    assert passed is False


def test_tester_fallback_no_keyword():
    """Non-JSON response without any keyword should default to False."""
    response = "Some ambiguous output"
    passed = "passed" in response.lower() and "fail" not in response.lower()
    assert passed is False
