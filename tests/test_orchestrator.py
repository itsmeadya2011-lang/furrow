import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def make_client(provider="anthropic"):
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider=provider)
    return client


def test_get_tasks_returns_tracked_tasks():
    client = make_client()
    orchestrator = Orchestrator("do stuff", client=client)
    tasks = [TaskModel(id="1", description="task one")]
    orchestrator._tasks = tasks
    assert orchestrator._get_tasks() == tasks


def test_is_done_no_tasks_returns_false():
    client = make_client()
    orchestrator = Orchestrator("do stuff", client=client)
    assert orchestrator._is_done() is False


def test_is_done_all_completed_and_tests_passed():
    client = make_client()
    orchestrator = Orchestrator("do stuff", client=client)
    orchestrator._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orchestrator._last_test_passed = True
    assert orchestrator._is_done() is True


def test_is_done_all_completed_but_tests_failed():
    client = make_client()
    orchestrator = Orchestrator("do stuff", client=client)
    orchestrator._tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    orchestrator._last_test_passed = False
    assert orchestrator._is_done() is False


def test_is_done_has_failed_tasks():
    client = make_client()
    orchestrator = Orchestrator("do stuff", client=client)
    orchestrator._tasks = [
        TaskModel(id="1", description="a", status="failed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orchestrator._last_test_passed = True
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_max_cycles_enforcement():
    client = make_client()
    client.complete = AsyncMock(side_effect=[
        json.dumps({"tasks": [{"id": "1", "description": "t1", "files": []}], "rationale": "ok"}),
        json.dumps({"files": {}, "summary": "done"}),
    ])
    client.write_file = AsyncMock()
    orchestrator = Orchestrator("do stuff", client=client)
    orchestrator.planner = PlannerAgent(client=client)

    with patch("furrow.core.orchestrator.settings") as mock_settings:
        mock_settings.max_cycles = 1
        with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester_run:
            mock_tester_run.return_value = TestResult(passed=True, summary="ok", failures=[])
            await orchestrator.run()

    assert orchestrator.cycles == 1


@pytest.mark.asyncio
async def test_cycle_flow_planning_execution_testing():
    client = make_client()
    plan_json = json.dumps({
        "tasks": [{"id": "1", "description": "write file", "files": ["out.txt"], "dependencies": []}],
        "rationale": "ok",
    })
    worker_json = json.dumps({
        "files": {"out.txt": "content here"},
        "summary": "wrote out.txt",
    })
    client.complete = AsyncMock(side_effect=[plan_json, worker_json])
    client.write_file = AsyncMock()

    orchestrator = Orchestrator("do stuff", client=client)
    orchestrator.planner = PlannerAgent(client=client)

    with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester_run:
        mock_tester_run.return_value = TestResult(passed=True, summary="tests ok", failures=[])
        await orchestrator._cycle()

    assert len(orchestrator._tasks) == 1
    assert orchestrator._tasks[0].status == "completed"
    assert orchestrator._last_test_passed is True
    client.write_file.assert_called_once_with("out.txt", "content here")
