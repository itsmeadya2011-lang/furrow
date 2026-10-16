import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.settings = MagicMock()
    client.settings.max_cycles = 0
    client.settings.planner_model = "planner-model"
    client.settings.worker_model = "worker-model"
    client.settings.tester_model = "tester-model"
    return client


@pytest.fixture
def orch(mock_client):
    return Orchestrator(goal="Build a feature", client=mock_client, console=MagicMock())


# --- Orchestrator: _is_done ---

def test_is_done_empty_tasks(orch):
    orch._tasks = []
    assert orch._is_done() is True


def test_is_done_all_completed(orch):
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_some_pending(orch):
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_is_done_failed_task(orch):
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_max_cycles_reached(mock_client):
    orch = Orchestrator(goal="Build a feature", client=mock_client, console=MagicMock())
    mock_client.settings.max_cycles = 2
    orch.cycles = 2
    orch._tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is True


# --- Orchestrator: _get_tasks ---

def test_get_tasks_returns_stored_tasks(orch):
    stored = [TaskModel(id="1", description="a")]
    orch._tasks = stored
    assert orch._get_tasks() is stored


# --- Orchestrator: _cycle ---

@patch("furrow.core.orchestrator.WorkerAgent")
@patch("furrow.core.orchestrator.TesterAgent")
@patch("furrow.core.orchestrator.Status", new_callable=MagicMock)
def test_cycle_stores_plan_tasks(mock_status, mock_tester, mock_worker, orch, mock_client):
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="write code", files=["a.py"]),
            TaskModel(id="2", description="write tests", files=["test_a.py"]),
        ],
        rationale="do it",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    mock_worker.return_value.run = AsyncMock(return_value="done")
    mock_tester.return_value.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    asyncio.run(orch._cycle())

    assert orch._tasks == plan.tasks
    assert len(orch._tasks) == 2
    mock_tester.return_value.run.assert_awaited_once_with(orch.goal, plan.tasks)


@patch("furrow.core.orchestrator.WorkerAgent")
@patch("furrow.core.orchestrator.TesterAgent")
@patch("furrow.core.orchestrator.Status", new_callable=MagicMock)
def test_task_status_updates(mock_status, mock_tester, mock_worker, orch, mock_client):
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", files=["a.py"]),
            TaskModel(id="2", description="b", files=["b.py"]),
        ],
        rationale="do it",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    mock_worker.return_value.run = AsyncMock(side_effect=["done", RuntimeError("boom")])
    mock_tester.return_value.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    asyncio.run(orch._cycle())

    assert orch._tasks[0].status == "completed"
    assert orch._tasks[0].result == "done"
    assert orch._tasks[1].status == "failed"
    assert "boom" in orch._tasks[1].result


@patch("furrow.core.orchestrator.WorkerAgent")
@patch("furrow.core.orchestrator.TesterAgent")
@patch("furrow.core.orchestrator.Status", new_callable=MagicMock)
def test_goal_rewriting_on_test_failure(mock_status, mock_tester, mock_worker, orch, mock_client):
    plan = Plan(
        tasks=[TaskModel(id="1", description="a", files=["a.py"])],
        rationale="do it",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    mock_worker.return_value.run = AsyncMock(return_value="done")
    mock_tester.return_value.run = AsyncMock(
        return_value=TestResult(
            passed=False, summary="failures", failures=["test_a", "test_b"]
        )
    )

    asyncio.run(orch._cycle())

    assert orch.goal == "Fix failing tests:\ntest_a\ntest_b"


# --- PlannerAgent ---

async def test_planner_successful_parse(mock_client):
    agent = PlannerAgent(client=mock_client)
    mock_client.complete = AsyncMock(
        return_value='{"tasks": [{"id": "1", "description": "implement x", "files": ["x.py"]}], "rationale": "because"}'
    )
    plan = await agent.plan("implement x")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "implement x"
    assert plan.rationale == "because"
    mock_client.complete.assert_awaited_once()
    args, kwargs = mock_client.complete.call_args
    assert kwargs["model"] == mock_client.settings.planner_model


async def test_planner_invalid_json_raises(mock_client):
    agent = PlannerAgent(client=mock_client)
    mock_client.complete = AsyncMock(return_value="this is not json")
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await agent.plan("do something")


async def test_planner_prompt_includes_goal(mock_client):
    agent = PlannerAgent(client=mock_client)
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "r"}')
    await agent.plan("my special goal")
    prompt = mock_client.complete.call_args[0][0]
    assert PLANNER_PROMPT in prompt
    assert "my special goal" in prompt


# --- WorkerAgent ---

async def test_worker_successful_run(mock_client):
    task = TaskModel(id="1", description="implement login", files=["auth.py", "models.py"])
    agent = WorkerAgent(task=task, client=mock_client)
    mock_client.complete = AsyncMock(return_value="login implemented")
    result = await agent.run()
    assert result == "login implemented"
    prompt = mock_client.complete.call_args[0][0]
    assert "implement login" in prompt
    assert "auth.py" in prompt
    assert "models.py" in prompt


# --- TesterAgent ---

async def test_tester_successful_parsing(mock_client):
    agent = TesterAgent(client=mock_client)
    agent._run_tests = AsyncMock(return_value="Ran 5 tests, 0 failed")
    mock_client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "all passed", "failures": []}'
    )
    result = await agent.run("goal", [TaskModel(id="1", description="t")])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "all passed"
    assert result.failures == []


async def test_tester_invalid_json_fallback(mock_client):
    agent = TesterAgent(client=mock_client)
    agent._run_tests = AsyncMock(return_value="some output")
    mock_client.complete = AsyncMock(return_value="All tests passed successfully")
    result = await agent.run("goal", [])
    assert result.passed is True
    assert result.summary == "All tests passed successfully"


async def test_tester_run_tests_exception(mock_client):
    agent = TesterAgent(client=mock_client)
    agent._run_tests = AsyncMock(side_effect=RuntimeError("crash"))
    result = await agent.run("goal", [])
    assert result.passed is False
    assert "crash" in result.summary
    assert result.failures == ["crash"]
