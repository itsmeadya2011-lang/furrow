import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.core.orchestrator import Orchestrator
from furrow.config import Plan, TaskModel


@pytest.mark.asyncio
async def test_get_tasks_empty_before_plan():
    client = MagicMock()
    orchestrator = Orchestrator(goal="test", client=client)
    assert orchestrator._get_tasks() == []


@pytest.mark.asyncio
async def test_get_tasks_after_plan():
    client = MagicMock()
    orchestrator = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator.current_plan = plan
    assert len(orchestrator._get_tasks()) == 1
    assert orchestrator._get_tasks()[0].description == "task 1"


@pytest.mark.asyncio
async def test_is_done_no_tasks():
    client = MagicMock()
    orchestrator = Orchestrator(goal="test", client=client)
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_is_done_all_completed():
    client = MagicMock()
    orchestrator = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator.current_plan = plan
    orchestrator.current_plan.tasks[0].status = "completed"
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_is_done_one_failed():
    client = MagicMock()
    orchestrator = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")], rationale="test")
    orchestrator.current_plan = plan
    orchestrator.current_plan.tasks[0].status = "completed"
    orchestrator.current_plan.tasks[1].status = "failed"
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_is_done_mixed_status():
    client = MagicMock()
    orchestrator = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test")
    orchestrator.current_plan = plan
    orchestrator.current_plan.tasks[0].status = "pending"
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_event_callback_invoked():
    client = MagicMock()
    messages = []
    def on_event(msg: str):
        messages.append(msg)
    orchestrator = Orchestrator(goal="test", client=client, on_event=on_event)
    orchestrator._output("hello")
    assert messages == ["hello"]


@pytest.mark.asyncio
async def test_max_cycles_stops():
    from furrow.agents.tester import TesterAgent
    original_run_tests = TesterAgent._run_tests
    TesterAgent._run_tests = AsyncMock(return_value="ok")
    try:
        client = MagicMock()
        client.settings.max_cycles = 2
        client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator.planner.plan = AsyncMock(return_value=Plan(tasks=[TaskModel(id="1", description="task 1")], rationale="test"))
        orchestrator._is_done = MagicMock(side_effect=[False, True])
        await orchestrator.run()
        assert orchestrator.cycles == 2
    finally:
        TesterAgent._run_tests = original_run_tests
