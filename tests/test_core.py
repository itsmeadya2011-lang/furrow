from __future__ import annotations

import json
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


def test_orchestrator_cycles_once_when_no_tasks():
    client = MagicMock(spec=LLMClient)
    client.settings.max_cycles = 0
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    with patch("furrow.core.orchestrator.PlannerAgent", return_value=planner):
        orch = Orchestrator(goal="do nothing", client=client)
        orch.cycles = 0
        assert orch._get_tasks() == []


def test_orchestrator_is_done_when_all_completed():
    client = MagicMock(spec=LLMClient)
    client.settings.max_cycles = 0
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    orch = Orchestrator(goal="x", client=client)
    orch.plan = plan
    assert orch._is_done() is True


def test_orchestrator_not_done_when_failed():
    client = MagicMock(spec=LLMClient)
    client.settings.max_cycles = 0
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    orch = Orchestrator(goal="x", client=client)
    orch.plan = plan
    assert orch._is_done() is False


def test_orchestrator_not_done_when_pending():
    client = MagicMock(spec=LLMClient)
    client.settings.max_cycles = 0
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="ok",
    )
    orch = Orchestrator(goal="x", client=client)
    orch.plan = plan
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_tester_parse_valid_json():
    client = MagicMock(spec=LLMClient)
    client.settings.tester_model = "test"
    client.complete = AsyncMock(return_value=json.dumps({"passed": True, "summary": "ok", "failures": []}))
    from furrow.agents.tester import TesterAgent
    t = TesterAgent(client=client)
    result = await t.run("goal", [])
    assert result.passed is True
    assert result.summary == "ok"


@pytest.mark.asyncio
async def test_tester_parse_invalid_json_falls_back():
    client = MagicMock(spec=LLMClient)
    client.settings.tester_model = "test"
    client.complete = AsyncMock(return_value="All tests passed")
    from furrow.agents.tester import TesterAgent
    t = TesterAgent(client=client)
    result = await t.run("goal", [])
    assert result.passed is True


@pytest.mark.asyncio
async def test_tester_parse_invalid_json_failure():
    client = MagicMock(spec=LLMClient)
    client.settings.tester_model = "test"
    client.complete = AsyncMock(return_value="Tests failed with errors")
    from furrow.agents.tester import TesterAgent
    t = TesterAgent(client=client)
    result = await t.run("goal", [])
    assert result.passed is False
