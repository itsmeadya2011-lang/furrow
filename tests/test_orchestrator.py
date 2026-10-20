"""Tests for the orchestrator: the plan -> execute -> test loop."""

from __future__ import annotations

import asyncio

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskStatus
from furrow.core.events import Event
from furrow.core.orchestrator import Orchestrator
from tests.helpers import FakeLLMClient, make_plan, make_test_result, make_worker

EVENT_KEYS: dict[str, set[str]] = {
    "cycle_start": {"cycle"},
    "plan": {"cycle", "tasks", "rationale"},
    "task_start": {"cycle", "task_id", "description"},
    "task_done": {"cycle", "task_id", "result", "files_changed"},
    "task_failed": {"cycle", "task_id", "error"},
    "test_result": {"cycle", "passed", "summary", "failures"},
    "cycle_end": {"cycle"},
    "done": {"cycles", "reason"},
}


class ConcurrencyProbe:
    """Records how many workers run at the same time."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.calls = 0

    async def run(self, worker: WorkerAgent) -> str:
        self.calls += 1
        self.current += 1
        self.peak = max(self.peak, self.current)
        await asyncio.sleep(0.01)
        self.current -= 1
        return f"finished {worker.task.id}"


async def test_orchestrator_keeps_cycling_while_the_tests_keep_failing(
    settings: Settings,
) -> None:
    settings.max_cycles = 3
    client = FakeLLMClient(
        settings,
        plans=[make_plan("step one"), make_plan("step two"), make_plan("step three")],
        workers=[make_worker(), make_worker(), make_worker()],
        tests=[make_test_result(False, "boom", ["boom"])] * 3,
    )
    events: list[Event] = []
    orch = Orchestrator("make it work", client=client, on_event=events.append)

    await orch.run()

    assert orch.cycles == 3
    assert len(orch.history) == 3
    assert sum(1 for event in events if event.kind == "cycle_start") == 3
    assert sum(1 for event in events if event.kind == "cycle_end") == 3


async def test_orchestrator_stops_at_max_cycles(settings: Settings) -> None:
    settings.max_cycles = 2
    client = FakeLLMClient(
        settings,
        plans=[make_plan("step one"), make_plan("step two")],
        workers=[make_worker(), make_worker()],
        tests=[make_test_result(False, "boom")] * 2,
    )
    events: list[Event] = []
    orch = Orchestrator("make it work", client=client, on_event=events.append)

    await orch.run()

    assert orch.cycles == 2
    done = [event for event in events if event.kind == "done"]
    assert len(done) == 1
    assert done[0].data["reason"] == "max cycles reached"
    assert done[0].data["cycles"] == 2


async def test_orchestrator_stops_when_the_goal_is_complete(settings: Settings) -> None:
    settings.max_cycles = 5
    client = FakeLLMClient(
        settings,
        plans=[make_plan("only task")],
        workers=[make_worker()],
        tests=[make_test_result(True, "all good")],
    )
    events: list[Event] = []
    orch = Orchestrator("make it work", client=client, on_event=events.append)

    await orch.run()

    assert orch.cycles == 1
    done = [event for event in events if event.kind == "done"][-1]
    assert done.data["reason"] == "goal complete"


async def test_orchestrator_respects_max_parallel_tasks(
    settings: Settings, monkeypatch
) -> None:
    settings.max_parallel_tasks = 2
    settings.max_cycles = 1
    probe = ConcurrencyProbe()

    async def fake_run(self: WorkerAgent) -> str:
        probe.calls += 1
        probe.current += 1
        probe.peak = max(probe.peak, probe.current)
        await asyncio.sleep(0.01)
        probe.current -= 1
        return f"finished {self.task.id}"

    monkeypatch.setattr(WorkerAgent, "run", fake_run)
    client = FakeLLMClient(
        settings,
        plans=[make_plan("one", "two", "three")],
        tests=[make_test_result(False, "boom")],
    )
    orch = Orchestrator("make it work", client=client)

    await orch.run()

    assert probe.calls == 3
    assert probe.peak == 2


async def test_orchestrator_marks_a_failing_worker_without_losing_its_siblings(
    settings: Settings, monkeypatch
) -> None:
    settings.max_cycles = 1

    async def fake_run(self: WorkerAgent) -> str:
        if self.task.id == "1":
            raise RuntimeError("worker exploded")
        await asyncio.sleep(0)
        return "ok"

    monkeypatch.setattr(WorkerAgent, "run", fake_run)
    client = FakeLLMClient(
        settings,
        plans=[make_plan("first", "second")],
        tests=[make_test_result(False, "boom")],
    )
    events: list[Event] = []
    orch = Orchestrator("make it work", client=client, on_event=events.append)

    await orch.run()

    failed = [event for event in events if event.kind == "task_failed"]
    done = [event for event in events if event.kind == "task_done"]
    assert len(failed) == 1
    assert failed[0].data["task_id"] == "1"
    assert "worker exploded" in failed[0].data["error"]
    assert len(done) == 1
    assert done[0].data["task_id"] == "2"
    statuses = {task.id: task.status for task in orch.tasks}
    assert statuses == {"1": TaskStatus.FAILED, "2": TaskStatus.COMPLETED}


async def test_orchestrator_emits_the_documented_events_in_order(settings: Settings) -> None:
    settings.max_cycles = 1
    client = FakeLLMClient(
        settings,
        plans=[make_plan("only task")],
        workers=[make_worker()],
        tests=[make_test_result(True, "all good")],
    )
    events: list[Event] = []
    orch = Orchestrator("make it work", client=client, on_event=events.append)

    await orch.run()

    assert [event.kind for event in events] == [
        "cycle_start",
        "plan",
        "task_start",
        "task_done",
        "test_result",
        "cycle_end",
        "done",
    ]
    for event in events:
        assert EVENT_KEYS[event.kind] <= set(event.data), event.kind


async def test_orchestrator_plan_event_carries_the_planned_tasks(settings: Settings) -> None:
    settings.max_cycles = 1
    client = FakeLLMClient(
        settings,
        plans=[make_plan("only task")],
        workers=[make_worker()],
        tests=[make_test_result(False, "boom")],
    )
    events: list[Event] = []
    orch = Orchestrator("make it work", client=client, on_event=events.append)

    await orch.run()

    plan_event = next(iter([event for event in events if event.kind == "plan"]))
    assert plan_event.data["cycle"] == 1
    assert plan_event.data["rationale"]
    assert plan_event.data["tasks"] == [
        {"id": "1", "description": "only task", "files": []}
    ]


async def test_orchestrator_narrows_the_goal_but_keeps_the_original(settings: Settings) -> None:
    settings.max_cycles = 1
    client = FakeLLMClient(
        settings,
        plans=[make_plan("only task")],
        workers=[make_worker()],
        tests=[make_test_result(False, "boom", ["tests/test_x.py::test_y"])],
    )
    orch = Orchestrator("build a parser", client=client)

    await orch.run()

    assert orch.original_goal == "build a parser"
    assert orch.goal != orch.original_goal
    assert "tests/test_x.py::test_y" in orch.goal
    assert "build a parser" in orch.goal


async def test_orchestrator_stops_when_it_stops_making_progress(
    settings: Settings, monkeypatch
) -> None:
    settings.max_cycles = 0
    settings.max_consecutive_failures = 2

    async def always_fail(self: WorkerAgent) -> str:
        raise RuntimeError("nope")

    monkeypatch.setattr(WorkerAgent, "run", always_fail)
    client = FakeLLMClient(
        settings,
        plans=[make_plan("first"), make_plan("second")],
        tests=[make_test_result(False, "boom")] * 2,
    )
    events: list[Event] = []
    orch = Orchestrator("make it work", client=client, on_event=events.append)

    await orch.run()

    assert orch.cycles == 2
    done = [event for event in events if event.kind == "done"][-1]
    assert done.data["reason"] == "no progress"


async def test_orchestrator_accumulates_tasks_across_cycles(settings: Settings) -> None:
    settings.max_cycles = 3
    client = FakeLLMClient(
        settings,
        plans=[
            make_plan("one", ids=["1"]),
            make_plan("two", ids=["2"]),
            make_plan("one again", ids=["1"]),
        ],
        workers=[make_worker(), make_worker(), make_worker()],
        tests=[make_test_result(False, "boom")] * 3,
    )
    orch = Orchestrator("make it work", client=client)

    await orch.run()

    assert [task.id for task in orch.tasks] == ["1", "2"]
    assert [entry["cycle"] for entry in orch.history] == [1, 2, 3]
    # merge-in-place: re-planning id "1" updates existing task, doesn't duplicate
    task_one = next(t for t in orch.tasks if t.id == "1")
    assert task_one.description == "one again"


async def test_orchestrator_survives_a_broken_event_listener(settings: Settings) -> None:
    settings.max_cycles = 1

    def broken_listener(event: Event) -> None:
        raise RuntimeError("listener exploded")

    client = FakeLLMClient(
        settings,
        plans=[make_plan("only task")],
        workers=[make_worker()],
        tests=[make_test_result(False, "boom")],
    )
    orch = Orchestrator("make it work", client=client, on_event=broken_listener)

    await orch.run()

    assert orch.cycles == 1
