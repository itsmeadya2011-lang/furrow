from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from furrow.config import Plan, TaskModel
from furrow.config import TestResult as ResultModel
from furrow.core import orchestrator as orchestrator_mod
from furrow.core.orchestrator import Orchestrator


class Recorder:
    """Shared, cross-task bookkeeping for the fake worker."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.concurrent = 0
        self.max_concurrent = 0

    def enter(self, task_id: str) -> None:
        self.events.append(("start", task_id))
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)

    def leave(self, task_id: str) -> None:
        self.events.append(("finish", task_id))
        self.concurrent -= 1

    def index(self, phase: str, task_id: str) -> int:
        return self.events.index((phase, task_id))

    def started_tasks(self) -> list[str]:
        return [tid for phase, tid in self.events if phase == "start"]


def make_plan(*tasks: TaskModel, rationale: str = "because") -> Plan:
    return Plan(tasks=list(tasks), rationale=rationale)


def install_fakes(monkeypatch, *, plans, recorder=None, results=None, planner_goals=None, fail_ids=(), delay=0.0):
    """Patch PlannerAgent/WorkerAgent/TesterAgent inside the orchestrator module."""
    plans = list(plans)
    goals = planner_goals if planner_goals is not None else []
    rec = recorder if recorder is not None else Recorder()
    results = list(results) if results is not None else [ResultModel(passed=True, summary="ok")]
    fail_ids = set(fail_ids)
    state = {"planner_calls": 0, "tester_calls": 0}

    class FakePlanner:
        def __init__(self, client=None, settings=None, **kwargs):
            self.client = client
            self.index = 0

        async def plan(self, goal):
            goals.append(goal)
            state["planner_calls"] += 1
            plan = plans[min(self.index, len(plans) - 1)]
            self.index += 1
            return plan

    class FakeWorker:
        def __init__(self, task=None, client=None, settings=None, **kwargs):
            self.task = task
            self.client = client

        async def run(self):
            rec.enter(self.task.id)
            try:
                if delay:
                    await asyncio.sleep(delay)
                if self.task.id in fail_ids:
                    raise RuntimeError(f"worker exploded on {self.task.id}")
                return "done"
            finally:
                rec.leave(self.task.id)

    class FakeTester:
        def __init__(self, client=None, settings=None, **kwargs):
            self.client = client

        async def run(self, goal, tasks):
            state["tester_calls"] += 1
            result = results[min(state["tester_calls"] - 1, len(results) - 1)]
            return result.model_copy(deep=True)

    monkeypatch.setattr(orchestrator_mod, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orchestrator_mod, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_mod, "TesterAgent", FakeTester)
    return rec, state, goals


def make_orchestrator(goal, settings, fake_llm_cls, on_event=None, client=None):
    client = client or fake_llm_cls(settings=settings)
    return Orchestrator(goal=goal, client=client, on_event=on_event)


def types_of(events):
    return [event["type"] for event in events]


def statuses(orchestrator):
    return {task.id: task.status for task in orchestrator.tasks}


# ----------------------------------------------------------------------
# run(): happy path
# ----------------------------------------------------------------------


async def test_run_completes_and_emits_lifecycle_events(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="a", description="first"),
        TaskModel(id="b", description="second", dependencies=["a"]),
    )
    install_fakes(monkeypatch, plans=[plan])
    events: list[dict] = []
    orchestrator = make_orchestrator(
        "ship it", settings_factory(), fake_llm_cls, on_event=events.append
    )

    await orchestrator.run()

    emitted = types_of(events)
    assert "cycle_start" in emitted
    assert "plan" in emitted
    assert "task_start" in emitted
    assert "task_complete" in emitted
    assert "test_result" in emitted
    assert "cycle_complete" in emitted
    assert emitted[-1] == "halted"
    assert events[-1]["reason"] == "complete"

    assert orchestrator.cycles == 1
    assert statuses(orchestrator) == {"a": "completed", "b": "completed"}
    assert all(task.result == "done" for task in orchestrator.tasks)
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is True
    assert orchestrator._is_done() is True

    plan_event = next(event for event in events if event["type"] == "plan")
    assert plan_event["task_ids"] == ["a", "b"]
    assert plan_event["rationale"] == "because"
    test_event = next(event for event in events if event["type"] == "test_result")
    assert test_event["passed"] is True
    assert test_event["summary"] == "ok"


async def test_run_halts_complete_when_planner_returns_no_tasks(monkeypatch, settings_factory, fake_llm_cls):
    _, state, _ = install_fakes(monkeypatch, plans=[make_plan(rationale="nothing left to do")])
    events: list[dict] = []
    orchestrator = make_orchestrator(
        "already done", settings_factory(), fake_llm_cls, on_event=events.append
    )

    await orchestrator.run()

    assert events[-1] == {"type": "halted", "reason": "complete", "cycles": 1}
    assert state["tester_calls"] == 0
    assert orchestrator.cycles == 1
    assert orchestrator.tasks == []
    assert "task_start" not in types_of(events)


# ----------------------------------------------------------------------
# Dependency waves
# ----------------------------------------------------------------------


async def test_dependency_waves_run_dependents_after_prerequisites(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="a", description="root"),
        TaskModel(id="b", description="needs a", dependencies=["a"]),
        TaskModel(id="c", description="independent root"),
    )
    recorder, _, _ = install_fakes(monkeypatch, plans=[plan])
    orchestrator = make_orchestrator("deps", settings_factory(), fake_llm_cls)

    await orchestrator.run()

    assert statuses(orchestrator) == {"a": "completed", "b": "completed", "c": "completed"}
    assert recorder.index("finish", "a") < recorder.index("start", "b")
    assert recorder.index("start", "a") < recorder.index("start", "b")
    assert recorder.index("start", "c") < recorder.index("start", "b")
    assert recorder.events[-2:] == [("start", "b"), ("finish", "b")]


# ----------------------------------------------------------------------
# Parallelism limits
# ----------------------------------------------------------------------


async def test_max_parallel_tasks_one_serialises_workers(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="a", description="one"),
        TaskModel(id="b", description="two"),
        TaskModel(id="c", description="three"),
        TaskModel(id="d", description="four"),
    )
    recorder, _, _ = install_fakes(monkeypatch, plans=[plan], delay=0.01)
    orchestrator = make_orchestrator(
        "serial", settings_factory(max_parallel_tasks=1), fake_llm_cls
    )

    await orchestrator.run()

    assert recorder.max_concurrent == 1
    assert sorted(recorder.started_tasks()) == ["a", "b", "c", "d"]
    assert statuses(orchestrator) == {"a": "completed", "b": "completed", "c": "completed", "d": "completed"}


async def test_parallel_slots_allow_concurrent_workers(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="a", description="one"),
        TaskModel(id="b", description="two"),
        TaskModel(id="c", description="three"),
    )
    recorder, _, _ = install_fakes(monkeypatch, plans=[plan], delay=0.01)
    orchestrator = make_orchestrator(
        "parallel", settings_factory(max_parallel_tasks=5), fake_llm_cls
    )

    await orchestrator.run()

    assert recorder.max_concurrent == 3


# ----------------------------------------------------------------------
# max_cycles
# ----------------------------------------------------------------------


async def test_max_cycles_halts_after_exactly_two_cycles(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(TaskModel(id="only", description="retry me"))
    failing = ResultModel(passed=False, summary="1 failed", failures=["tests/test_x.py::test_y"])
    recorder, state, goals = install_fakes(monkeypatch, plans=[plan], results=[failing])
    events: list[dict] = []
    orchestrator = make_orchestrator(
        "keep failing",
        settings_factory(max_cycles=2),
        fake_llm_cls,
        on_event=events.append,
    )

    await orchestrator.run()

    assert state["planner_calls"] == 2
    assert state["tester_calls"] == 2
    assert types_of(events).count("cycle_complete") == 2
    assert events[-1]["type"] == "halted"
    assert events[-1]["reason"] == "max_cycles"
    assert recorder.events == [("start", "only"), ("finish", "only")] * 2
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is False
    assert orchestrator.fix_requests == ["tests/test_x.py::test_y", "tests/test_x.py::test_y"]
    assert "Outstanding failures to fix" not in goals[0]
    assert "Outstanding failures to fix" in goals[1]
    assert "tests/test_x.py::test_y" in goals[1]
    assert statuses(orchestrator) == {"only": "completed"}


# ----------------------------------------------------------------------
# State persistence
# ----------------------------------------------------------------------


async def test_state_file_written_and_orchestrator_resumes(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="a", description="first"),
        TaskModel(id="b", description="second", dependencies=["a"]),
    )
    install_fakes(monkeypatch, plans=[plan])
    settings = settings_factory()
    state_file = Path(settings.state_file)
    orchestrator = make_orchestrator("resume me", settings, fake_llm_cls)

    await orchestrator.run()

    assert state_file.exists()
    saved = json.loads(state_file.read_text())
    assert saved["cycles"] == 1
    assert [task["id"] for task in saved["tasks"]] == ["a", "b"]
    assert [task["status"] for task in saved["tasks"]] == ["completed", "completed"]
    assert saved["last_test_result"]["passed"] is True
    assert saved["goal"] == "resume me"

    resumed = make_orchestrator("resume me", settings, fake_llm_cls)
    assert resumed.cycles == 1
    assert [task.id for task in resumed.tasks] == ["a", "b"]
    assert statuses(resumed) == {"a": "completed", "b": "completed"}
    assert [task.result for task in resumed.tasks] == ["done", "done"]
    assert resumed.last_test_result is not None
    assert resumed.last_test_result.passed is True
    assert resumed._is_done() is True


async def test_state_saved_for_incomplete_run(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(TaskModel(id="a", description="only"))
    failing = ResultModel(passed=False, summary="nope", failures=["boom"])
    install_fakes(monkeypatch, plans=[plan], results=[failing])
    settings = settings_factory(max_cycles=1)
    state_file = Path(settings.state_file)
    orchestrator = make_orchestrator("unfinished", settings, fake_llm_cls)

    await orchestrator.run()

    saved = json.loads(state_file.read_text())
    assert saved["cycles"] == 1
    assert saved["complete"] is False
    assert saved["last_test_result"]["passed"] is False
    assert saved["fix_requests"] == ["boom"]

    resumed = make_orchestrator("unfinished", settings, fake_llm_cls)
    assert resumed.cycles == 1
    assert resumed.last_test_result is not None
    assert resumed.last_test_result.passed is False
    assert resumed.fix_requests == ["boom"]
    assert resumed._is_done() is False


def test_corrupt_state_file_is_ignored(monkeypatch, settings_factory, fake_llm_cls):
    install_fakes(monkeypatch, plans=[make_plan()])
    settings = settings_factory()
    state_file = Path(settings.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("{not json", encoding="utf-8")

    orchestrator = make_orchestrator("fresh start", settings, fake_llm_cls)

    assert orchestrator.cycles == 0
    assert orchestrator.tasks == []
    assert orchestrator.last_test_result is None


# ----------------------------------------------------------------------
# _is_done
# ----------------------------------------------------------------------


def test_is_done_false_without_tasks(monkeypatch, settings_factory, fake_llm_cls):
    install_fakes(monkeypatch, plans=[make_plan()])
    orchestrator = make_orchestrator("nothing", settings_factory(), fake_llm_cls)

    assert orchestrator.tasks == []
    assert orchestrator._is_done() is False


def test_is_done_true_when_tasks_completed_and_tests_passed(monkeypatch, settings_factory, fake_llm_cls):
    install_fakes(monkeypatch, plans=[make_plan()])
    orchestrator = make_orchestrator("done", settings_factory(), fake_llm_cls)
    orchestrator.tasks = [
        TaskModel(id="a", description="x", status="completed", result="done"),
        TaskModel(id="b", description="y", status="completed", result="done"),
    ]
    orchestrator.last_test_result = ResultModel(passed=True, summary="ok")

    assert orchestrator._is_done() is True


def test_is_done_false_when_a_task_failed(monkeypatch, settings_factory, fake_llm_cls):
    install_fakes(monkeypatch, plans=[make_plan()])
    orchestrator = make_orchestrator("broken", settings_factory(), fake_llm_cls)
    orchestrator.tasks = [
        TaskModel(id="a", description="x", status="completed", result="done"),
        TaskModel(id="b", description="y", status="failed", result="boom"),
    ]
    orchestrator.last_test_result = ResultModel(passed=True, summary="ok")

    assert orchestrator._is_done() is False


@pytest.mark.parametrize("status", ["pending", "running", "blocked", "failed"])
def test_is_done_false_for_unfinished_statuses(monkeypatch, settings_factory, fake_llm_cls, status):
    install_fakes(monkeypatch, plans=[make_plan()])
    orchestrator = make_orchestrator("unfinished", settings_factory(), fake_llm_cls)
    orchestrator.tasks = [TaskModel(id="a", description="x", status=status)]
    orchestrator.last_test_result = ResultModel(passed=True, summary="ok")

    assert orchestrator._is_done() is False


def test_is_done_false_without_passing_test_result(monkeypatch, settings_factory, fake_llm_cls):
    install_fakes(monkeypatch, plans=[make_plan()])
    orchestrator = make_orchestrator("no tests yet", settings_factory(), fake_llm_cls)
    orchestrator.tasks = [TaskModel(id="a", description="x", status="completed", result="done")]

    assert orchestrator._is_done() is False
    orchestrator.last_test_result = ResultModel(passed=False, summary="bad")
    assert orchestrator._is_done() is False


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------


async def test_failed_task_blocks_dependent_tasks(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="a", description="root"),
        TaskModel(id="b", description="will explode"),
        TaskModel(id="c", description="needs b", dependencies=["b"]),
    )
    recorder, _, _ = install_fakes(monkeypatch, plans=[plan], fail_ids={"b"})
    events: list[dict] = []
    orchestrator = make_orchestrator(
        "failure", settings_factory(), fake_llm_cls, on_event=events.append
    )
    orchestrator.cycles = 1

    await orchestrator._cycle()

    assert statuses(orchestrator) == {"a": "completed", "b": "failed", "c": "blocked"}
    tasks = {task.id: task for task in orchestrator.tasks}
    assert "worker exploded on b" in tasks["b"].result
    assert tasks["c"].result is not None
    assert "b" in tasks["c"].result
    assert recorder.started_tasks() == ["a", "b"]

    failed = [event for event in events if event["type"] == "task_failed"]
    assert [event["task_id"] for event in failed] == ["b"]
    assert failed[0]["error"] == "worker exploded on b"

    blocked = [event for event in events if event["type"] == "task_blocked"]
    assert [event["task_id"] for event in blocked] == ["c"]
    assert blocked[0]["blocked_by"] == ["b"]

    started_ids = [event["task_id"] for event in events if event["type"] == "task_start"]
    assert "c" not in started_ids
    assert orchestrator._is_done() is False


async def test_run_retries_failed_tasks_until_max_cycles(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="b", description="will explode"),
        TaskModel(id="c", description="needs b", dependencies=["b"]),
    )
    failing = ResultModel(passed=False, summary="1 failed", failures=["boom"])
    recorder, state, _ = install_fakes(
        monkeypatch, plans=[plan], results=[failing], fail_ids={"b"}
    )
    events: list[dict] = []
    orchestrator = make_orchestrator(
        "never green",
        settings_factory(max_cycles=2),
        fake_llm_cls,
        on_event=events.append,
    )

    await orchestrator.run()

    assert state["planner_calls"] == 2
    assert recorder.events == [("start", "b"), ("finish", "b")] * 2
    assert types_of(events).count("task_blocked") == 2
    assert statuses(orchestrator) == {"b": "failed", "c": "blocked"}
    assert events[-1]["reason"] == "max_cycles"
    assert orchestrator.fix_requests == ["boom", "boom"]


async def test_unresolvable_dependencies_fail_remaining_tasks(monkeypatch, settings_factory, fake_llm_cls):
    plan = make_plan(
        TaskModel(id="a", description="depends on b", dependencies=["b"]),
        TaskModel(id="b", description="depends on a", dependencies=["a"]),
    )
    recorder, _, _ = install_fakes(monkeypatch, plans=[plan])
    events: list[dict] = []
    orchestrator = make_orchestrator(
        "cycle", settings_factory(), fake_llm_cls, on_event=events.append
    )
    orchestrator._merge_tasks(plan.tasks)

    await orchestrator._execute(plan.tasks)

    assert statuses(orchestrator) == {"a": "failed", "b": "failed"}
    assert recorder.events == []
    failed = [event for event in events if event["type"] == "task_failed"]
    assert sorted(event["task_id"] for event in failed) == ["a", "b"]
    assert all(event["error"] == "Unresolvable dependencies" for event in failed)
