from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.fixture
def sample_plan_json() -> str:
    return json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "Implement feature A", "files": ["src/a.py"]},
                {
                    "id": "2",
                    "description": "Add tests",
                    "files": ["tests/test_a.py"],
                    "dependencies": ["1"],
                },
            ],
            "rationale": "Break goal into implementation + tests.",
        }
    )


def test_planner_agent_parse(sample_plan_json: str):
    client = MagicMock()
    client.complete = AsyncMock(return_value=sample_plan_json)
    client.settings = MagicMock()
    client.settings.planner_model = "claude-3-5-haiku-20241022"

    agent = PlannerAgent(client=client)
    plan = asyncio.run(agent.plan("implement feature A"))

    assert len(plan.tasks) == 2
    assert plan.tasks[0].description == "Implement feature A"
    assert plan.rationale == "Break goal into implementation + tests."
    client.complete.assert_awaited_once()


def test_planner_agent_invalid_json():
    client = MagicMock()
    client.complete = AsyncMock(return_value="not valid json {{{")
    client.settings = MagicMock()
    client.settings.planner_model = "claude-3-5-haiku-20241022"

    agent = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        asyncio.run(agent.plan("some goal"))


def test_worker_agent_returns_summary():
    client = MagicMock()
    client.complete = AsyncMock(return_value="Implemented feature A and saved files.")
    client.settings = MagicMock()
    client.settings.worker_model = "claude-3-5-sonnet-20241022"

    task = TaskModel(id="1", description="Implement feature A")
    agent = WorkerAgent(task=task, client=client)
    result = asyncio.run(agent.run())

    assert isinstance(result, str)
    assert "Implemented feature A" in result
    client.complete.assert_awaited_once()


def test_tester_agent_pass():
    client = MagicMock()
    test_result_json = json.dumps({"passed": True, "summary": "All tests passed", "failures": []})
    client.complete = AsyncMock(return_value=test_result_json)
    client.settings = MagicMock()
    client.settings.tester_model = "claude-3-5-sonnet-20241022"

    agent = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="task 1")]

    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(b"test output", b""))
    proc.kill = MagicMock()

    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", return_value=proc):
        result = asyncio.run(agent.run("do thing", tasks))

    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


def test_tester_agent_fail():
    client = MagicMock()
    test_result_json = json.dumps(
        {"passed": False, "summary": "2 tests failed", "failures": ["test_one", "test_two"]}
    )
    client.complete = AsyncMock(return_value=test_result_json)
    client.settings = MagicMock()
    client.settings.tester_model = "claude-3-5-sonnet-20241022"

    agent = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="task 1")]

    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(b"FAILURES", b""))
    proc.kill = MagicMock()

    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", return_value=proc):
        result = asyncio.run(agent.run("do thing", tasks))

    assert result.passed is False
    assert result.summary == "2 tests failed"
    assert result.failures == ["test_one", "test_two"]
    client.complete.assert_awaited_once()
