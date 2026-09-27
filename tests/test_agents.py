from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult


async def test_planner_agent_parse_valid():
    mock_client = MagicMock()
    plan_data = {
        "tasks": [
            {
                "id": "1",
                "description": "Implement auth",
                "files": ["src/auth.py"],
                "dependencies": [],
            }
        ],
        "rationale": "Add authentication",
    }
    mock_client.complete = AsyncMock(return_value=json.dumps(plan_data))

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("test goal")

    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Implement auth"
    assert plan.rationale == "Add authentication"


async def test_planner_agent_parse_invalid():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="not valid json")

    planner = PlannerAgent(client=mock_client)

    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("test goal")


async def test_worker_agent_run():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="Done with task")

    task = TaskModel(id="1", description="do thing", files=["src/file.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Done with task"


async def test_worker_agent_empty_files():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="Done with task")

    task = TaskModel(id="1", description="do thing", files=[])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Done with task"


async def test_tester_agent_no_test_runner():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "All good", "failures": []}'
    )

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="No test runner found.")

    result = await tester.run("test goal", [])

    assert isinstance(result, TestResult)
    assert result.passed is True


async def test_tester_agent_json_response():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "All tests passed", "failures": []}'
    )

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="test output here")

    result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


async def test_tester_agent_json_response_failed():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value='{"passed": false, "summary": "Tests failed", "failures": ["test_1 failed"]}'
    )

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="test output here")

    result = await tester.run("test goal", [])

    assert result.passed is False
    assert "test_1 failed" in result.failures
