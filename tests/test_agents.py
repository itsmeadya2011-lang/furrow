import json
from unittest.mock import AsyncMock, Mock

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient


def _make_mock_client():
    client = Mock(spec=LLMClient)
    client.settings = Mock()
    return client


async def test_worker_returns_summary():
    mock_client = _make_mock_client()
    mock_client.settings.worker_model = "worker-model"
    mock_client.complete = AsyncMock(return_value="Summary of changes")

    task = TaskModel(id="1", description="do thing")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Summary of changes"
    mock_client.complete.assert_awaited_once()


async def test_tester_no_runner_found():
    mock_client = _make_mock_client()
    mock_client.settings.tester_model = "tester-model"
    mock_client.complete = AsyncMock(
        return_value=json.dumps({"passed": True, "summary": "All tests passed", "failures": []})
    )

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="No test runner found.")

    result = await tester.run("goal", [])

    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


async def test_tester_handles_test_exception():
    mock_client = _make_mock_client()
    mock_client.settings.tester_model = "tester-model"

    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(side_effect=OSError("test runner error"))

    result = await tester.run("goal", [])

    assert result.passed is False
    assert "test runner error" in result.summary


async def test_planner_parses_json():
    plan_json = json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "Implement feature X", "files": ["src/x.py"], "dependencies": []}
            ],
            "rationale": "Need to implement X",
        }
    )
    mock_client = _make_mock_client()
    mock_client.settings.planner_model = "planner-model"
    mock_client.complete = AsyncMock(return_value=plan_json)

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Implement feature X")

    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Implement feature X"
    assert plan.rationale == "Need to implement X"


async def test_planner_raises_on_bad_json():
    mock_client = _make_mock_client()
    mock_client.settings.planner_model = "planner-model"
    mock_client.complete = AsyncMock(return_value="not json")

    planner = PlannerAgent(client=mock_client)

    try:
        await planner.plan("goal")
        assert False, "Expected ValueError"
    except ValueError:
        pass
