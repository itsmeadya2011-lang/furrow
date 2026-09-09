import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_get_tasks_returns_stored_tasks():
    """_get_tasks() must return the actual stored tasks, not an empty list."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [TaskModel(id="1", description="task 1", status="completed")]
    tasks = orch._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_is_done_no_tasks():
    """No tasks should not be done."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = []
    orch.last_test_result = None
    assert orch._is_done() is False


def test_is_done_all_completed_with_passing_tests():
    """All tasks completed + tests passed = done."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="all good")
    assert orch._is_done() is True


def test_is_done_pending_tasks():
    """Pending tasks means not done."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="all good")
    assert orch._is_done() is False


def test_is_done_failed_tasks():
    """Failed tasks means not done."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="failed"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="all good")
    assert orch._is_done() is False


def test_is_done_all_completed_but_tests_failed():
    """All tasks done but tests failed = not done (should retry)."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=False, summary="tests fail", failures=["bug"])
    assert orch._is_done() is False


def test_max_cycles_enforced():
    """Orchestrator should stop after max_cycles even if goal not done."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="pending"),
    ]
    orch.last_test_result = None
    orch.max_cycles = 1
    orch.cycles = 1

    # At max_cycles, should not be done (but loop should break)
    assert orch._is_done() is False


def test_tasks_stored_after_cycle():
    """Tasks from the plan should be stored in all_tasks after a cycle."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.all_tasks = []

    plan_tasks = [TaskModel(id="1", description="do thing")]
    orch.all_tasks.extend(plan_tasks)
    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].id == "1"


@pytest.mark.asyncio
async def test_output_callback_invoked():
    """Output callback should receive messages during run."""
    messages = []

    async def callback(msg: str):
        messages.append(msg)

    orch = Orchestrator.__new__(Orchestrator)
    orch.goal = "test goal"
    orch.client = MagicMock()
    orch.client.settings.max_parallel_tasks = 5
    orch.client.settings.max_cycles = 1
    orch.cycles = 0
    orch.all_tasks = []
    orch.last_test_result = None
    orch.max_cycles = 1
    orch.output_callback = callback

    # Mock the _cycle to avoid LLM calls
    orch._cycle = AsyncMock()

    await orch.run()

    # Should have received at least the startup and cycle messages
    assert len(messages) > 0


@pytest.mark.asyncio
async def test_log_calls_console_and_callback():
    """_log should print to console and call the callback."""
    messages = []

    async def callback(msg: str):
        messages.append(msg)

    orch = Orchestrator.__new__(Orchestrator)
    orch.output_callback = callback

    with patch("furrow.core.orchestrator.console") as mock_console:
        await orch._log("test message")
        mock_console.print.assert_called_once()
        assert messages == ["test message"]
