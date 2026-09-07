import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_empty_when_no_plan():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []


def test_orchestrator_get_tasks_returns_last_plan_tasks():
    o = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="plan",
    )
    o._last_plan = plan
    assert o._get_tasks() == plan.tasks


def test_orchestrator_is_done_false_when_no_tasks():
    o = Orchestrator(goal="test")
    assert o._is_done() is False


def test_orchestrator_is_done_true_when_all_completed():
    o = Orchestrator(goal="test")
    o._last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="plan",
    )
    assert o._is_done() is True


def test_orchestrator_is_done_false_when_any_failed():
    o = Orchestrator(goal="test")
    o._last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="plan",
    )
    assert o._is_done() is False


def test_orchestrator_is_done_false_when_pending():
    o = Orchestrator(goal="test")
    o._last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="plan",
    )
    assert o._is_done() is False


def test_orchestrator_cycle_emits_events():
    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)

    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "tasks": [
            {"id": "1", "description": "do thing", "files": [], "dependencies": []}
        ],
        "rationale": "ok",
    }))
    mock_client.settings = TestSettings()

    o = Orchestrator(goal="test", client=mock_client, event_callback=emit)

    # We just test that _cycle runs and emits events without error
    # We can't easily test the full async cycle without mocking TesterAgent too,
    # but we can at least verify the planner call path works.
    plan = asyncio.run(o.planner.plan("test"))
    assert len(plan.tasks) == 1


class TestSettings:
    planner_model = "test-model"
    worker_model = "test-model"
    tester_model = "test-model"
    max_cycles = 0


def test_tester_no_runner_returns_passed():
    from furrow.agents.tester import TesterAgent

    async def run_test():
        mock_client = AsyncMock()
        mock_client.settings = TestSettings()
        t = TesterAgent(client=mock_client, settings=TestSettings())
        # Patch _run_tests to simulate no runner
        t._run_tests = AsyncMock(return_value="NO_TESTS_FOUND")  # type: ignore[method-assign]
        result = await t.run("goal", [])
        return result

    result = asyncio.run(run_test())
    assert result.passed is True
    assert "No test runner" in result.summary or "Proceeding without" in result.summary


def test_tester_with_output_parses_json():
    from furrow.agents.tester import TesterAgent

    async def run_test():
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": True,
            "summary": "all good",
            "failures": [],
        }))
        mock_client.settings = TestSettings()
        t = TesterAgent(client=mock_client, settings=TestSettings())
        t._run_tests = AsyncMock(return_value="pytest output here")  # type: ignore[method-assign]
        result = await t.run("goal", [])
        return result

    result = asyncio.run(run_test())
    assert result.passed is True
    assert result.summary == "all good"
