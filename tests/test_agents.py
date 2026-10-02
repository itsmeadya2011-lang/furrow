from __future__ import annotations

import json

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from tests.conftest import FakeClient

VALID_PLAN = json.dumps(
    {
        "tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}],
        "rationale": "r",
    }
)


def _task() -> TaskModel:
    return TaskModel(id="1", description="x", files=[], dependencies=[])


async def test_planner_plan_parses_valid():
    client = FakeClient(response=VALID_PLAN)
    plan = await PlannerAgent(client=client).plan("build a thing")
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "r"


async def test_planner_plan_invalid_json_raises():
    client = FakeClient(response="not json at all")
    with pytest.raises(ValueError):
        await PlannerAgent(client=client).plan("goal")


async def test_planner_plan_strips_markdown_fences():
    fenced = "```json\n" + VALID_PLAN + "\n```"
    client = FakeClient(response=fenced)
    plan = await PlannerAgent(client=client).plan("goal")
    assert len(plan.tasks) == 1


async def test_worker_run_returns_result():
    client = FakeClient(response="done")
    result = await WorkerAgent(task=_task(), client=client).run()
    assert result == "done"


async def test_tester_run_with_json_result(monkeypatch):
    payload = json.dumps({"passed": False, "summary": "fail", "failures": ["x"]})
    client = FakeClient(response=payload)
    tester = TesterAgent(client=client)

    async def fake_run_tests():
        return "some output"

    monkeypatch.setattr(tester, "_run_tests", fake_run_tests)
    result = await tester.run("goal", [_task()])
    assert isinstance(result, TestResult)
    assert result.passed is False
    assert result.summary == "fail"
    assert result.failures == ["x"]


async def test_tester_run_fallback_no_json(monkeypatch):
    client = FakeClient(response="all good, passed")
    tester = TesterAgent(client=client)

    async def fake_run_tests():
        return "output"

    monkeypatch.setattr(tester, "_run_tests", fake_run_tests)
    result = await tester.run("goal", [_task()])
    assert result.passed is True
    assert "all good" in result.summary
