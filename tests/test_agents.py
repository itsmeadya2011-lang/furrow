import json
from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient


@pytest.fixture
def settings():
    from furrow.config import Settings

    return Settings()


@pytest.fixture
def client(settings):
    c = LLMClient(settings=settings)
    c.complete = AsyncMock()
    return c


async def test_planner_parses_valid_json(client):
    client.complete.return_value = json.dumps(
        {"tasks": [{"id": "1", "description": "x"}], "rationale": "r"}
    )
    plan = await PlannerAgent(client=client).plan("goal")
    assert isinstance(plan, Plan)
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "r"


async def test_planner_invalid_json_raises(client):
    client.complete.return_value = "not json"
    with pytest.raises(ValueError):
        await PlannerAgent(client=client).plan("goal")


async def test_worker_returns_llm_text(client):
    client.complete.return_value = "worker-output"
    task = TaskModel(id="1", description="do x")
    result = await WorkerAgent(task=task, client=client).run()
    assert result == "worker-output"


async def test_tester_parses_passed_json(client):
    client.complete.return_value = json.dumps(
        {"passed": True, "summary": "ok", "failures": []}
    )
    result = await TesterAgent(client=client).run("goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "ok"


async def test_tester_substring_fallback_passed(client, monkeypatch):
    async def fake_run_tests(self):
        return "fake test output"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)
    client.complete.return_value = "passed everything"
    result = await TesterAgent(client=client).run("goal", [])
    assert result.passed is True


async def test_tester_substring_fallback_failed(client, monkeypatch):
    async def fake_run_tests(self):
        return "fake test output"

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)
    client.complete.return_value = "tests failed: ..."
    result = await TesterAgent(client=client).run("goal", [])
    assert result.passed is False


async def test_tester_no_runner_returns_failure(client, monkeypatch):
    async def fake_run_tests(self):
        return "No test runner found."

    monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)
    # Force substring fallback (invalid JSON). The response contains no
    # "passed" substring, so the tester should mark it as failed and surface
    # the no-runner message in the summary.
    client.complete.return_value = "no test runner found, nothing to do"
    result = await TesterAgent(client=client).run("goal", [])
    assert result.passed is False
    assert "No test runner found." in result.summary