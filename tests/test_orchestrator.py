from __future__ import annotations

import asyncio

import pytest

from furrow.agents.planner import Plan
from furrow.agents.tester import TestResult
from furrow.agents.worker import WorkerResult
from furrow.config import RunResult, Settings, TaskModel
from furrow.core.events import EventType
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore


class FakeWorker:
    def __init__(self, task, client=None, settings=None, workspace=None, goal=""):
        self.task = task
        self._order = None

    async def run(self):
        return WorkerResult(summary=f"did {self.task.id}", success=True, files_changed=[])


class FakePlanner:
    def __init__(self, tasks=None, raise_valueerror=False):
        self.tasks = tasks or [TaskModel(id="t1", description="do t1")]
        self.raise_valueerror = raise_valueerror
        self.calls = 0

    async def plan(self, goal, context=""):
        self.calls += 1
        if self.raise_valueerror:
            raise ValueError("boom")
        return Plan(
            tasks=[
                TaskModel(id=t.id, description=t.description, dependencies=list(t.dependencies))
                for t in self.tasks
            ],
            rationale="r",
        )


class FakeTester:
    def __init__(self, passed=False):
        self.passed = passed

    async def run(self, goal, tasks):
        return TestResult(passed=self.passed, summary="s", failures=["fail"])


class BoomConsole:
    """A console whose print always fails, like a broken terminal."""

    def print(self, *args, **kwargs):
        raise RuntimeError("boom")


class RecordingConsole:
    """A console that records every message instead of printing it."""

    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):
        self.messages.append(" ".join(self._plain(arg) for arg in args))

    @staticmethod
    def _plain(text):
        from rich.markup import render

        return "".join(str(part) for part in render(str(text)))


def make_settings(tmp_path, max_cycles=3, max_parallel_tasks=2):
    return Settings(_env_file=None, workspace=tmp_path, max_cycles=max_cycles,
                    max_parallel_tasks=max_parallel_tasks)


@pytest.mark.asyncio
async def test_loops_three_cycles(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=3)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)
    result = await orch.run()
    assert result.cycles == 3


@pytest.mark.asyncio
async def test_original_goal_survives(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="ORIG", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)
    await orch.run()
    assert orch.original_goal == "ORIG"
    assert "fail" in orch.goal


@pytest.mark.asyncio
async def test_tasks_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_dependency_ordering(tmp_path, monkeypatch):
    order = []

    class OW(FakeWorker):
        async def run(self):
            order.append(self.task.id)
            return WorkerResult(summary=f"did {self.task.id}", success=True, files_changed=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", OW)
    t1 = TaskModel(id="1", description="one")
    t2 = TaskModel(id="2", description="two", dependencies=["1"])
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner(tasks=[t1, t2])
    orch.tester = FakeTester(passed=True)
    await orch.run()
    assert order.index("1") < order.index("2")


@pytest.mark.asyncio
async def test_failed_dep_blocked(tmp_path, monkeypatch):
    class FW(FakeWorker):
        async def run(self):
            if self.task.id == "1":
                self.task.status = "failed"
                return WorkerResult(summary="fail", success=False, files_changed=[])
            return WorkerResult(summary="ok", success=True, files_changed=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    t1 = TaskModel(id="1", description="one")
    t2 = TaskModel(id="2", description="two", dependencies=["1"])
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner(tasks=[t1, t2])
    orch.tester = FakeTester(passed=True)
    await orch.run()
    t2o = next(t for t in orch.tasks if t.id == "2")
    assert t2o.status == "blocked"


@pytest.mark.asyncio
async def test_max_parallel(tmp_path, monkeypatch):
    maxc = {"v": 0}
    cur = {"v": 0}

    class SW(FakeWorker):
        async def run(self):
            cur["v"] += 1
            maxc["v"] = max(maxc["v"], cur["v"])
            await asyncio.sleep(0.05)
            cur["v"] -= 1
            return WorkerResult(summary=f"did {self.task.id}", success=True, files_changed=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", SW)
    tasks = [TaskModel(id=f"t{i}", description=f"d{i}") for i in range(4)]
    settings = make_settings(tmp_path, max_cycles=1, max_parallel_tasks=2)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner(tasks=tasks)
    orch.tester = FakeTester(passed=True)
    await orch.run()
    assert maxc["v"] <= 2


@pytest.mark.asyncio
async def test_no_progress_guard(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=10)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)
    result = await orch.run()
    assert result.stopped_reason == "no progress"


@pytest.mark.asyncio
async def test_state_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    assert store.exists()
    st = store.load()
    assert st.original_goal == "g"
    assert st.cycles > 0


@pytest.mark.asyncio
async def test_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    orch2 = Orchestrator(goal="x", settings=settings, state_store=store, resume=True)
    assert orch2.cycles > 0
    assert orch2.original_goal == "g"


@pytest.mark.asyncio
async def test_events(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    q = orch.event_bus.subscribe()
    await orch.run()
    types = []
    while not q.empty():
        types.append(q.get_nowait().type)
    assert EventType.CYCLE_STARTED in types
    assert EventType.TASK_STARTED in types
    assert EventType.TASK_FINISHED in types


@pytest.mark.asyncio
async def test_planner_valueerror(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner(raise_valueerror=True)
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    assert result.stopped_reason.startswith("planner failed")


@pytest.mark.asyncio
async def test_console_output_does_not_break_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    monkeypatch.setattr(orch, "_console", lambda: BoomConsole())
    result = await orch.run()
    assert isinstance(result, RunResult)
    assert result.cycles == 1
    assert result.stopped_reason == "max_cycles reached"


@pytest.mark.asyncio
async def test_status_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    status = orch.status()
    assert set(status) >= {"goal", "original_goal", "cycles", "stopped_reason", "tasks"}
    assert status["original_goal"] == "g"
    assert status["cycles"] == 1
    assert status["stopped_reason"] == "max_cycles reached"
    assert isinstance(status["tasks"], list)
    assert status["tasks"]
    assert all(isinstance(task, dict) for task in status["tasks"])


@pytest.mark.asyncio
async def test_console_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    console = RecordingConsole()
    monkeypatch.setattr(orch, "_console", lambda: console)
    await orch.run()
    printed = "\n".join(console.messages)
    assert "Goal: g" in printed
    assert "═══ Cycle 1 ═══" in printed
    assert "Plan: r (1 task)" in printed
    assert "Task t1 completed" in printed
    assert "Tests passed: s" in printed
    assert "Stopped: max_cycles reached" in printed


@pytest.mark.asyncio
async def test_console_reports_failed_task(tmp_path, monkeypatch):
    class FW(FakeWorker):
        async def run(self):
            return WorkerResult(summary="nope", success=False, files_changed=[], error="boom")

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)
    console = RecordingConsole()
    monkeypatch.setattr(orch, "_console", lambda: console)
    await orch.run()
    printed = "\n".join(console.messages)
    assert "Task t1 failed: boom" in printed
    assert "Tests failed: s" in printed
