import json

import pytest
from unittest.mock import AsyncMock

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeLLMClient:
    """Minimal LLMClient stand-in that returns canned JSON by prompt type."""

    def __init__(self, settings=None, planner_response=None, tester_response=None,
                 worker_response=None):
        self.settings = settings or Settings()
        self.planner_response = planner_response
        self.tester_response = tester_response
        self.worker_response = worker_response
        self.write_file_calls = []
        self.complete_calls = []

    async def complete(self, prompt, system="", model=None):
        self.complete_calls.append((prompt, system, model))
        low = prompt.lower()
        if "planning agent" in low or "break the user's goal" in low:
            return self.planner_response
        if "tester agent" in low:
            return self.tester_response
        if "worker agent" in low:
            return self.worker_response
        return "{}"

    async def read_file(self, path):
        return ""

    async def write_file(self, path, content):
        self.write_file_calls.append((str(path), content))

    def list_files(self, directory):
        return []


def _plan_json(tasks=None, rationale="ok"):
    return json.dumps({
        "tasks": tasks or [{"id": "1", "description": "do thing", "files": ["out.txt"]}],
        "rationale": rationale,
    })


def _tester_json(passed, summary="done", failures=None):
    return json.dumps({
        "passed": passed,
        "summary": summary,
        "failures": failures if failures is not None else [],
    })


def _worker_json(edits=None, summary="done"):
    return json.dumps({
        "edits": edits or [{"path": "out.txt", "action": "create", "content": "hi"}],
        "summary": summary,
    })


@pytest.mark.asyncio
async def test_orchestrator_loop_completes_when_all_tasks_done(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", AsyncMock(return_value=""))
    settings = Settings(max_cycles=0)
    client = FakeLLMClient(
        settings=settings,
        planner_response=_plan_json(),
        tester_response=_tester_json(True, "all good"),
        worker_response=_worker_json(),
    )
    orch = Orchestrator(goal="create out.txt", client=client)
    await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_loops_on_failed_tests(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", AsyncMock(return_value=""))
    monkeypatch.setattr(WorkerAgent, "run", AsyncMock(side_effect=RuntimeError("worker failed")))
    settings = Settings(max_cycles=2)
    client = FakeLLMClient(
        settings=settings,
        planner_response=_plan_json(),
        tester_response=_tester_json(False, "broken", failures=["boom"]),
        worker_response=_worker_json(),
    )
    orch = Orchestrator(goal="do thing", client=client)
    await orch.run()
    assert orch.cycles == 2
    assert orch.goal.startswith("Fix failing tests:")


@pytest.mark.asyncio
async def test_orchestrator_enforces_max_cycles(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", AsyncMock(return_value=""))
    settings = Settings(max_cycles=1)
    client = FakeLLMClient(
        settings=settings,
        planner_response=_plan_json(),
        tester_response=_tester_json(False, "broken", failures=["boom"]),
        worker_response=_worker_json(),
    )
    orch = Orchestrator(goal="do thing", client=client)
    await orch.run()
    assert orch.cycles == 1


def test_orchestrator_get_tasks_persists():
    settings = Settings()
    orch = Orchestrator(goal="x", client=FakeLLMClient(settings=settings))
    tasks = [TaskModel(id="1", description="x", status="completed")]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


@pytest.mark.asyncio
async def test_worker_writes_files():
    settings = Settings()
    client = FakeLLMClient(
        settings=settings,
        worker_response=_worker_json(
            edits=[
                {"path": "a.txt", "action": "create", "content": "aaa"},
                {"path": "b.txt", "action": "create", "content": "bbb"},
            ],
            summary="created two files",
        ),
    )
    task = TaskModel(id="1", description="create files", files=["a.txt", "b.txt"])
    agent = WorkerAgent(task=task, client=client)
    await agent.run()
    assert len(client.write_file_calls) == 2
    written = [call[0] for call in client.write_file_calls]
    assert written[0].endswith("a.txt")
    assert written[1].endswith("b.txt")
    assert client.write_file_calls[0][1] == "aaa"
    assert client.write_file_calls[1][1] == "bbb"


@pytest.mark.asyncio
async def test_tester_parses_json(monkeypatch):
    monkeypatch.setattr(TesterAgent, "_run_tests", AsyncMock(return_value=""))
    client = FakeLLMClient(
        settings=Settings(),
        tester_response=_tester_json(True, "all good", failures=[]),
    )
    agent = TesterAgent(client=client)
    result = await agent.run("my goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []