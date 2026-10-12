from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient, strip_code_fences
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_strip_code_fences():
    assert strip_code_fences("plain text") == "plain text"
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('  \n```json\n{"a": 1}\n```\n  ') == '{"a": 1}'


class _FakeLLMClientFenced(LLMClient):
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return '```json\n{"passed": true}\n```'


class _FakeLLMClientBadJson(LLMClient):
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return "not json"


async def test_complete_json_parses_fenced():
    client = _FakeLLMClientFenced(settings=Settings())
    result = await client.complete_json("prompt")
    assert result == {"passed": True}

    client_bad = _FakeLLMClientBadJson(settings=Settings())
    with pytest.raises(ValueError):
        await client_bad.complete_json("prompt")


class _FakePlannerClient:
    def __init__(self) -> None:
        self.settings = Settings()

    async def complete_json(self, prompt: str, model: str | None = None) -> dict:
        return {
            "tasks": [
                {"id": "1", "description": "task 1"},
                {"description": "task 2"},
                {"description": "task 3"},
                {"description": "task 4"},
                {"description": "task 5"},
                {"description": "task 6"},
                {"description": "task 7"},
            ],
            "rationale": "test",
        }


async def test_planner_clamps_and_normalizes():
    client = _FakePlannerClient()
    planner = PlannerAgent(client=client)
    plan = await planner.plan("test goal")

    assert len(plan.tasks) == 5
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].id == "2"
    for task in plan.tasks:
        assert task.files == []
        assert task.dependencies == []


class _FakePlannerClientEmpty:
    def __init__(self) -> None:
        self.settings = Settings()

    async def complete_json(self, prompt: str, model: str | None = None) -> dict:
        return {"tasks": [], "rationale": "test"}


async def test_planner_raises_on_empty_tasks():
    client = _FakePlannerClientEmpty()
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await planner.plan("test goal")


class _FakeWorkerClient:
    def __init__(self) -> None:
        self.settings = Settings()

    async def read_file(self, path: str) -> str:
        return "content of " + path

    async def complete_json(self, prompt: str, model: str | None = None) -> dict:
        return {
            "summary": "did it",
            "complete": True,
            "edits": [{"path": "a.py", "content": "new"}],
        }


async def test_worker_returns_edits_json():
    client = _FakeWorkerClient()
    task = TaskModel(id="1", description="d", files=["a.py"])
    result_json = await WorkerAgent(task=task, client=client).run()
    data = json.loads(result_json)
    assert data["summary"] == "did it"
    assert data["complete"] is True
    assert data["edits"] == [{"path": "a.py", "content": "new"}]


class _FakeTesterClientBadJson:
    def __init__(self) -> None:
        self.settings = Settings()

    async def complete_json(self, prompt: str, model: str | None = None) -> dict:
        # Mirror real LLMClient.complete_json, which embeds the raw
        # response in the error message.
        raise ValueError("Failed to parse JSON response: ... Raw response: 'tests passed ok'")

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return "tests passed ok"


async def test_tester_fallback_on_bad_json(monkeypatch):
    async def _stub_run_tests(self) -> str:
        return "stub test output"

    monkeypatch.setattr(TesterAgent, "_run_tests", _stub_run_tests)

    client = _FakeTesterClientBadJson()
    tester = TesterAgent(client=client)
    result = await tester.run("goal", [])
    assert result.passed is True
    assert "passed" in result.summary.lower()


class _FakeOrchestratorClient:
    def __init__(self) -> None:
        self.settings = Settings()


async def test_orchestrator_is_done_scenarios():
    # (a) no tasks -> True
    orch = Orchestrator(goal="g", client=_FakeOrchestratorClient())
    assert orch._is_done() is True

    # (b) one failed task -> False
    orch2 = Orchestrator(goal="g", client=_FakeOrchestratorClient())
    orch2.tasks = [TaskModel(id="1", description="d", status="failed")]
    assert orch2._is_done() is False

    # (c) all completed, last_test_result None -> True
    orch3 = Orchestrator(goal="g", client=_FakeOrchestratorClient())
    orch3.tasks = [TaskModel(id="1", description="d", status="completed")]
    assert orch3._is_done() is True

    # (d) all completed, TestResult(passed=False) -> False
    orch4 = Orchestrator(goal="g", client=_FakeOrchestratorClient())
    orch4.tasks = [TaskModel(id="1", description="d", status="completed")]
    orch4.last_test_result = TestResult(passed=False, summary="fail", failures=[])
    assert orch4._is_done() is False


class _FakeExecutorClient:
    def __init__(self) -> None:
        self.settings = Settings()
        self.written: list[tuple[str, str]] = []

    async def write_file(self, path: Path | str, content: str) -> None:
        p = Path(path)
        self.written.append((str(p), content))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except OSError:
            pass  # absolute path outside the sandbox; recording is enough


class _StubWorkerAgent:
    def __init__(self, task: TaskModel, client: Any = None, settings: Any = None) -> None:
        self.task = task
        self.client = client
        self.call_count = 0

    async def run(self) -> str:
        self.call_count += 1
        if self.call_count == 1:
            return '{"summary": "ok", "complete": true, "edits": [{"path": "out.txt", "content": "hello"}]}'
        else:
            # Second call: absolute path edit
            return '{"summary": "ok", "complete": true, "edits": [{"path": "/abs/out2.txt", "content": "world"}]}'


async def test_orchestrator_execute_task_applies_edits(monkeypatch, tmp_path: Path):
    import furrow.core.orchestrator as orch_module

    client = _FakeExecutorClient()
    client.settings.workspace = tmp_path

    monkeypatch.setattr(orch_module, "WorkerAgent", _StubWorkerAgent)

    orch = Orchestrator(goal="g", client=client)
    task = TaskModel(id="1", description="d", files=[])

    await orch._execute_task(task)

    assert (tmp_path / "out.txt").read_text() == "hello"
    assert task.status == "completed"

    # Second task with absolute path
    task2 = TaskModel(id="2", description="d2", files=[])
    await orch._execute_task(task2)

    # Absolute path must not be joined with the workspace
    assert ("/abs/out2.txt", "world") in client.written
    assert not (tmp_path / "abs" / "out2.txt").exists()


async def test_orchestrator_emit_callbacks():
    orch = Orchestrator(goal="g", client=_FakeOrchestratorClient())
    sync_events: list[dict] = []
    async_events: list[dict] = []

    def sync_cb(event: dict) -> None:
        sync_events.append(event)

    async def async_cb(event: dict) -> None:
        async_events.append(event)

    orch.on_event.append(sync_cb)
    orch.on_event.append(async_cb)

    orch._emit({"type": "x"})

    assert sync_events == [{"type": "x"}]
    # async callback is scheduled via create_task; yield control
    await asyncio.sleep(0)
    assert async_events == [{"type": "x"}]