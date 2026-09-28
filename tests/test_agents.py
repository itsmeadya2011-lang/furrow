from __future__ import annotations

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests._fakes import (
    FakeLLMClient,
    FakeSettings,
    fake_run_tests_ok,
    fake_run_tests_raise,
    make_plan_json,
    make_test_result,
)


@pytest.mark.asyncio
async def test_planner_parses_plan():
    client = FakeLLMClient(plans=[make_plan_json()], settings=FakeSettings())
    plan = await PlannerAgent(client=client).plan("build auth")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "ok"


@pytest.mark.asyncio
async def test_planner_raises_on_bad_json():
    client = FakeLLMClient(plans=["not json"], settings=FakeSettings())
    with pytest.raises(ValueError):
        await PlannerAgent(client=client).plan("build auth")


@pytest.mark.asyncio
async def test_worker_returns_summary():
    client = FakeLLMClient(workers=["added login route"], settings=FakeSettings())
    task = TaskModel(id="1", description="add login", files=[], dependencies=[])
    result = await WorkerAgent(task=task, client=client).run()
    assert "login route" in result


@pytest.mark.asyncio
async def test_tester_parses_pass(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_ok)
    client = FakeLLMClient(
        test_results=[make_test_result(passed=True, summary="all good")],
        settings=FakeSettings(),
    )
    res = await TesterAgent(client=client).run("goal", [])
    assert res.passed is True
    assert res.summary == "all good"


@pytest.mark.asyncio
async def test_tester_unparseable_response(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_ok)
    client = FakeLLMClient(test_results=["no json here"], settings=FakeSettings())
    res = await TesterAgent(client=client).run("goal", [])
    assert res.passed is False
    assert res.summary == "no json here"


@pytest.mark.asyncio
async def test_tester_handles_runner_error(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests_raise)
    client = FakeLLMClient(test_results=[], settings=FakeSettings())
    res = await TesterAgent(client=client).run("goal", [])
    assert res.passed is False
    assert res.summary == "simulated test runner failure"
