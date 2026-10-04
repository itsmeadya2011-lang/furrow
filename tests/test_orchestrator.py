from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from furrow.config import (
    FileEdit,
    Plan,
    Settings,
    TaskModel,
    TestResult,
    WorkerResult,
)
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator


class FakeLLM:
    def __init__(self, settings):
        self.settings = settings
        self.written: dict[Path, str] = {}

    async def write_file(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[p] = content


class FakePlanner:
    queue: list[Plan] = []
    calls: list[tuple[str, str]] = []

    def __init__(self, client=None, settings=None):
        self.client = client
        self.settings = settings

    async def plan(self, goal, context=""):
        self.calls.append((goal, context))
        return self.queue.pop(0)


class FakeWorker:
    active = 0
    max_active = 0
    count = 0

    def __init__(self, task=None, client=None, settings=None):
        self.task = task
        self.client = client
        self.settings = settings

    async def run(self):
        FakeWorker.active += 1
        FakeWorker.max_active = max(FakeWorker.max_active, FakeWorker.active)
        await asyncio.sleep(0.01)
        FakeWorker.active -= 1
        FakeWorker.count += 1
        return WorkerResult(
            summary=f"did {self.task.id}",
            edits=[FileEdit(path="out.txt", content=f"task {self.task.id}\n")],
        )


class FakeWorkerRaising:
    def __init__(self, task=None, client=None, settings=None):
        self.task = task
        self.client = client
        self.settings = settings

    async def run(self):
        raise RuntimeError("boom")


class FakeTester:
    queue: list[TestResult] = []
    calls: list[tuple[str, str]] = []

    def __init__(self, client=None, settings=None):
        self.client = client
        self.settings = settings

    async def run(self, goal, tasks, context=""):
        self.calls.append((goal, context))
        return self.queue.pop(0)


@pytest.fixture(autouse=True)
def reset_fakes():
    FakePlanner.queue = []
    FakePlanner.calls = []
    FakeTester.queue = []
    FakeTester.calls = []
    FakeWorker.active = 0
    FakeWorker.max_active = 0
    FakeWorker.count = 0
    yield


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", FakeTester)


def _names(events):
    return [name for name, _ in events]


def _contains_in_order(names, expected):
    it = iter(names)
    return all(any(n == e for n in it) for e in expected)


def _task(id="1"):
    return TaskModel(id=id, description=f"task {id}", files=["a.py"])


async def test_single_passing_cycle(tmp_path, patched):
    events = []
    on_event = lambda e, d: events.append((e, d))
    FakePlanner.queue = [Plan(tasks=[_task()], rationale="r")]
    FakeTester.queue = [TestResult(passed=True, summary="ok", failures=[])]
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path),
        on_event=on_event,
    )
    await orchestrator.run()
    names = _names(events)
    assert _contains_in_order(
        names,
        ["start", "cycle", "plan", "task_start", "edit", "task_done", "test", "done"],
    )
    assert (tmp_path / "out.txt").read_text() == "task 1\n"
    assert orchestrator._is_done() is True


async def test_goal_preserved_across_cycles(tmp_path, patched):
    events = []
    on_event = lambda e, d: events.append((e, d))
    FakePlanner.queue = [
        Plan(tasks=[_task()], rationale="r1"),
        Plan(tasks=[_task()], rationale="r2"),
    ]
    FakeTester.queue = [
        TestResult(passed=False, summary="bad", failures=["f"]),
        TestResult(passed=True, summary="ok", failures=[]),
    ]
    orchestrator = Orchestrator(
        goal="original goal",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path),
        on_event=on_event,
    )
    await orchestrator.run()
    assert orchestrator.cycles == 2
    assert len(FakeTester.calls) == 2
    assert FakeTester.calls[0][0] == "original goal"
    assert FakeTester.calls[1][0] == "original goal"
    assert len(FakePlanner.calls) == 2
    assert "tests failed" in FakePlanner.calls[1][1]


async def test_max_cycles_stops(tmp_path, patched):
    events = []
    on_event = lambda e, d: events.append((e, d))
    FakePlanner.queue = [Plan(tasks=[_task()], rationale="r")]
    FakeTester.queue = [TestResult(passed=False, summary="bad", failures=[])]
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path, max_cycles=1),
        on_event=on_event,
    )
    await orchestrator.run()
    assert ("stopped", {"cycles": 1, "reason": "max_cycles"}) in events
    assert orchestrator.cycles == 1


async def test_empty_plan_completes(tmp_path, patched):
    events = []
    on_event = lambda e, d: events.append((e, d))
    FakePlanner.queue = [Plan(tasks=[], rationale="done")]
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path),
        on_event=on_event,
    )
    await orchestrator.run()
    assert FakeTester.calls == []
    assert any(name == "done" for name, _ in events)


async def test_worker_failure_marks_task_failed(tmp_path, patched, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorkerRaising)
    events = []
    on_event = lambda e, d: events.append((e, d))
    FakePlanner.queue = [Plan(tasks=[_task()], rationale="r")]
    FakeTester.queue = [TestResult(passed=True, summary="ok", failures=[])]
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path, max_cycles=1),
        on_event=on_event,
    )
    await orchestrator.run()
    task_done_events = [(name, data) for name, data in events if name == "task_done"]
    assert task_done_events
    assert task_done_events[0][1]["status"] == "failed"
    assert orchestrator._is_done() is False
    assert len(FakeTester.calls) == 1


async def test_apply_edit_refuses_escape(tmp_path, patched):
    events = []
    on_event = lambda e, d: events.append((e, d))
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path),
        on_event=on_event,
    )
    await orchestrator._apply_edit(FileEdit(path="../escape.txt", content="x"))
    assert any(name == "error" for name, _ in events)
    assert not (tmp_path.parent / "escape.txt").exists()

    await orchestrator._apply_edit(FileEdit(path="pkg/mod.py", content="y"))
    assert (tmp_path / "pkg" / "mod.py").read_text() == "y"


async def test_parallelism_capped(tmp_path, patched):
    events = []
    on_event = lambda e, d: events.append((e, d))
    FakePlanner.queue = [Plan(tasks=[_task(str(i)) for i in range(5)], rationale="r")]
    FakeTester.queue = [TestResult(passed=True, summary="ok", failures=[])]
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path, max_parallel_tasks=2),
        on_event=on_event,
    )
    await orchestrator.run()
    assert FakeWorker.max_active <= 2
    assert FakeWorker.count == 5


async def test_context_truncation(tmp_path, patched):
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path),
    )
    orchestrator.history = ["x" * 100] * 100
    assert len(orchestrator._build_context()) == 4000


async def test_broken_on_event_does_not_kill_loop(tmp_path, patched):
    def broken(e, d):
        raise RuntimeError("boom")

    FakePlanner.queue = [Plan(tasks=[], rationale="done")]
    orchestrator = Orchestrator(
        goal="g",
        client=FakeLLM(Settings(workspace=tmp_path)),
        settings=Settings(workspace=tmp_path),
        on_event=broken,
    )
    await orchestrator.run()
    assert orchestrator.cycles == 1
