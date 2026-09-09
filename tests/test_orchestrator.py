from unittest.mock import AsyncMock, Mock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_get_tasks_returns_current_tasks():
    orchestrator = Orchestrator(goal="test goal", client=Mock(), max_cycles=1)
    tasks = [TaskModel(id="1", description="first task")]
    orchestrator.current_tasks = tasks

    assert orchestrator._get_tasks() is tasks


def test_is_done_returns_false_when_there_are_no_tasks():
    orchestrator = Orchestrator(goal="test goal", client=Mock(), max_cycles=1)

    assert orchestrator._is_done() is False


def test_is_done_returns_false_when_tasks_are_pending():
    orchestrator = Orchestrator(goal="test goal", client=Mock(), max_cycles=1)
    orchestrator.current_tasks = [TaskModel(id="1", description="pending task")]

    assert orchestrator._is_done() is False


def test_is_done_returns_false_when_any_task_failed():
    orchestrator = Orchestrator(goal="test goal", client=Mock(), max_cycles=1)
    orchestrator.current_tasks = [
        TaskModel(id="1", description="completed task", status="completed"),
        TaskModel(id="2", description="failed task", status="failed"),
    ]
    orchestrator.tests_passed = True

    assert orchestrator._is_done() is False


def test_is_done_returns_true_when_all_tasks_completed_and_tests_passed():
    orchestrator = Orchestrator(goal="test goal", client=Mock(), max_cycles=1)
    orchestrator.current_tasks = [
        TaskModel(id="1", description="first task", status="completed"),
        TaskModel(id="2", description="second task", status="completed"),
    ]
    orchestrator.tests_passed = True

    assert orchestrator._is_done() is True


def test_is_done_returns_false_when_all_tasks_completed_but_tests_failed():
    orchestrator = Orchestrator(goal="test goal", client=Mock(), max_cycles=1)
    orchestrator.current_tasks = [
        TaskModel(id="1", description="first task", status="completed"),
        TaskModel(id="2", description="second task", status="completed"),
    ]
    orchestrator.tests_passed = False

    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_run_halts_when_max_cycles_is_reached():
    on_output = Mock()
    orchestrator = Orchestrator(
        goal="test goal",
        client=Mock(),
        max_cycles=2,
        on_output=on_output,
    )
    orchestrator._cycle = AsyncMock()
    orchestrator._is_done = Mock(return_value=False)

    await orchestrator.run()

    assert orchestrator._cycle.await_count == 2
    assert orchestrator.cycles == 2
    on_output.assert_any_call("[bold yellow]Max cycles (2) reached. Halting.[/bold yellow]")


@pytest.mark.asyncio
async def test_on_output_callback_is_called():
    on_output = Mock()
    orchestrator = Orchestrator(
        goal="test goal",
        client=Mock(),
        max_cycles=1,
        on_output=on_output,
    )
    orchestrator._cycle = AsyncMock()
    orchestrator._is_done = Mock(return_value=True)

    await orchestrator.run()

    on_output.assert_called()


@pytest.mark.asyncio
async def test_cycle_updates_current_tasks_across_cycles(tmp_path):
    first_tasks = [TaskModel(id="1", description="first task")]
    second_tasks = [TaskModel(id="2", description="second task")]
    client = Mock(settings=Settings(workspace=tmp_path))
    orchestrator = Orchestrator(goal="test goal", client=client, max_cycles=2)
    orchestrator.planner.plan = AsyncMock(
        side_effect=[
            Plan(tasks=first_tasks, rationale="first plan"),
            Plan(tasks=second_tasks, rationale="second plan"),
        ]
    )
    worker = Mock()
    worker.run = AsyncMock(return_value="completed")
    tester = Mock()
    tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="tests passed", failures=[])
    )

    with (
        patch("furrow.core.orchestrator.WorkerAgent", return_value=worker) as worker_agent,
        patch("furrow.core.orchestrator.TesterAgent", return_value=tester) as tester_agent,
    ):
        await orchestrator._cycle()
        assert orchestrator.current_tasks == first_tasks
        assert all(task.status == "completed" for task in first_tasks)

        await orchestrator._cycle()
        assert orchestrator.current_tasks == second_tasks
        assert all(task.status == "completed" for task in second_tasks)

    assert orchestrator.planner.plan.await_count == 2
    assert worker.run.await_count == 2
    assert tester.run.await_count == 2
    assert worker_agent.call_count == 2
    assert tester_agent.call_count == 2
