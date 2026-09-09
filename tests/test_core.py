import asyncio
from pathlib import Path

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class _FakeLLMClient:
    """Minimal LLMClient stand-in for testing the orchestrator loop logic.

    Planner/worker/tester calls are routed through explicit methods so the
    fake can return deterministic responses per phase.
    """

    def __init__(self):
        self.plan_responses: list[str] = []
        self.test_responses: list[str] = []
        self.worker_responses: list[str] = []
        self.plan_calls = 0
        self.test_calls = 0
        self.worker_calls = 0

    @property
    def settings(self):
        from furrow.config import Settings
        return Settings()

    async def complete(self, prompt, system="", model=None):
        # The agents prefix their prompts with the stable prompt text from
        # furrow.agents.prompts: "You are a planning agent", "You are a worker
        # agent", "You are a tester agent". Dispatch on those markers.
        if "You are a planning agent" in prompt:
            return await self._plan()
        if "You are a tester agent" in prompt:
            return await self._test()
        return await self._worker()

    async def _plan(self):
        self.plan_calls += 1
        return self.plan_responses.pop(0) if self.plan_responses else '{"tasks": [], "rationale": "nothing"}'

    async def _test(self):
        self.test_calls += 1
        return self.test_responses.pop(0) if self.test_responses else '{"passed": true, "summary": "ok", "failures": []}'

    async def _worker(self):
        self.worker_calls += 1
        return self.worker_responses.pop(0) if self.worker_responses else '{"files": [], "summary": "done"}'

    async def read_file(self, path):
        raise FileNotFoundError(path)

    async def write_file(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list_files(self, directory):
        return []


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_halts_when_no_tasks():
    """If the planner returns no tasks, the loop should stop after one cycle."""
    orch = Orchestrator(goal="do nothing")
    orch.tasks = []
    assert orch._is_done() is True


def test_orchestrator_continues_on_failure():
    """Failed tasks should keep the loop going so fixes can be attempted."""
    orch = Orchestrator(goal="fix stuff")
    orch.tasks = [TaskModel(id="1", description="failing task", status="failed")]
    assert orch._is_done() is False


def test_orchestrator_completes_when_all_done():
    """All tasks completed -> loop should stop."""
    orch = Orchestrator(goal="done")
    orch.tasks = [TaskModel(id="1", description="ok", status="completed")]
    assert orch._is_done() is True


def test_orchestrator_runs_full_cycle(tmp_path, monkeypatch):
    """End-to-end: planner returns one task, worker writes a file, tests pass."""
    monkeypatch.chdir(tmp_path)
    client = _FakeLLMClient()
    client.plan_responses = [
        '{"tasks": [{"id": "1", "description": "create file", "files": ["out.txt"]}], "rationale": "one task"}'
    ]
    client.worker_responses = [
        '{"files": [{"path": "out.txt", "content": "hello"}], "summary": "wrote file"}'
    ]
    client.test_responses = ['{"passed": true, "summary": "all good", "failures": []}']
    orch = Orchestrator(goal="create out.txt", client=client)
    asyncio.run(orch.run())
    assert orch.cycles == 1
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_orchestrator_retries_on_failure(tmp_path, monkeypatch):
    """When tests fail, the goal is updated and the loop runs another cycle."""
    monkeypatch.chdir(tmp_path)
    client = _FakeLLMClient()
    client.plan_responses = [
        '{"tasks": [{"id": "1", "description": "write file", "files": ["f.txt"]}], "rationale": "first"}',
        '{"tasks": [{"id": "2", "description": "fix file", "files": ["f.txt"]}], "rationale": "fix"}',
    ]
    client.worker_responses = [
        '{"files": [{"path": "f.txt", "content": "x"}], "summary": "wrote"}',
        '{"files": [{"path": "f.txt", "content": "y"}], "summary": "fixed"}',
    ]
    client.test_responses = [
        '{"passed": false, "summary": "broken", "failures": ["test_foo failed"]}',
        '{"passed": true, "summary": "ok", "failures": []}',
    ]
    orch = Orchestrator(goal="write f.txt", client=client)
    asyncio.run(orch.run())
    assert orch.cycles == 2
    assert "Fix failing tests" in orch.goal
    assert (tmp_path / "f.txt").read_text() == "y"