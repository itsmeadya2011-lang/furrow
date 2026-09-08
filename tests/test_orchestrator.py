import json
from pathlib import Path

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class MockClient:
    def __init__(self):
        self.planner_model = "planner-model"
        self.settings = type("Settings", (), {"planner_model": "planner-model"})()

    async def complete(self, prompt, model=None, **kwargs):
        return json.dumps({
            "tasks": [{"id": "1", "description": "task"}],
            "rationale": "ok",
        })


class MockTester:
    def __init__(self, result=None):
        self.result = result or TestResult(passed=True, summary="ok", failures=[])

    async def run(self, goal, tasks):
        return self.result


class TestOrchestratorIsDone:
    def test_all_completed(self):
        o = Orchestrator.__new__(Orchestrator)
        o._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert o._is_done() is True

    def test_no_tasks(self):
        o = Orchestrator.__new__(Orchestrator)
        o._current_tasks = []
        assert o._is_done() is True

    def test_some_pending(self):
        o = Orchestrator.__new__(Orchestrator)
        o._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ]
        assert o._is_done() is False

    def test_failed_task(self):
        o = Orchestrator.__new__(Orchestrator)
        o._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ]
        assert o._is_done() is False

    def test_all_failed(self):
        o = Orchestrator.__new__(Orchestrator)
        o._current_tasks = [
            TaskModel(id="1", description="a", status="failed"),
            TaskModel(id="2", description="b", status="failed"),
        ]
        assert o._is_done() is False


class TestOrchestratorGetTasks:
    def test_returns_tracked_tasks(self):
        o = Orchestrator.__new__(Orchestrator)
        tasks = [TaskModel(id="1", description="a")]
        o._current_tasks = tasks
        assert o._get_tasks() is tasks


class TestOrchestratorMaxCycles:
    @pytest.mark.asyncio
    async def test_max_cycles_enforced(self, monkeypatch, capsys):
        o = Orchestrator(goal="test")

        async def fake_cycle(self):
            pass

        monkeypatch.setattr(Orchestrator, "_cycle", fake_cycle)
        monkeypatch.setattr(Orchestrator, "_is_done", lambda self: False)
        o.settings.max_cycles = 2
        o.settings.workspace = Path(".")

        await o.run()
        assert o.cycles == 2
