from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def _make_client(response: str | list[str]) -> AsyncMock:
    client = AsyncMock()
    if isinstance(response, list):
        client.complete = AsyncMock(side_effect=response)
    else:
        client.complete = AsyncMock(return_value=response)
    # Agents index `.settings.planner_model` etc.
    client.settings = type("S", (), {
        "planner_model": "pm",
        "worker_model": "wm",
        "tester_model": "tm",
    })()
    return client


@pytest.mark.asyncio
async def test_planner_parses_json() -> None:
    plan_data = {
        "tasks": [{"id": "t1", "description": "do a"}],
        "rationale": "because",
    }
    client = _make_client(json.dumps(plan_data))
    planner = PlannerAgent(client=client)
    plan = await planner.plan("build it")
    assert plan.rationale == "because"
    assert plan.tasks[0].id == "t1"


@pytest.mark.asyncio
async def test_planner_raises_on_bad_json() -> None:
    client = _make_client("not json")
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("goal")


@pytest.mark.asyncio
async def test_planner_appends_failure_context() -> None:
    plan_data = {"tasks": [], "rationale": "x"}
    client = _make_client(json.dumps(plan_data))
    planner = PlannerAgent(client=client)
    await planner.plan("goal", failure_context="tests failed")
    prompt_arg = client.complete.await_args.args[0]
    assert "tests failed" in prompt_arg
    assert "Failure context from previous cycle" in prompt_arg


@pytest.mark.asyncio
async def test_worker_returns_string() -> None:
    client = _make_client("ok")
    task = TaskModel(id="t1", description="write code")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "ok"


@pytest.mark.asyncio
async def test_tester_parses_json() -> None:
    result_data = {"passed": True, "summary": "all good", "failures": []}
    client = _make_client(json.dumps(result_data))
    tester = TesterAgent(client=client)

    # Patch out subprocess execution by short-circuiting _run_tests.
    tester._run_tests = AsyncMock(return_value="pytest: 1 passed")

    tr = await tester.run("goal", [TaskModel(id="t1", description="x")])
    assert tr.passed is True
    assert tr.summary == "all good"


@pytest.mark.asyncio
async def test_tester_fallback() -> None:
    client = _make_client("garbage that does not mention passed")
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="pytest: 1 failed")

    tr = await tester.run("goal", [])
    assert tr.passed is False
    assert tr.summary == "garbage that does not mention passed"
    assert tr.failures == []