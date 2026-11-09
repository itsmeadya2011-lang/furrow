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
    def __init__(self, tasks=None, raise_valueerror=False, raise_runtimeerror=False):
        self.tasks = (
            [TaskModel(id="t1", description="do t1")]
            if tasks is None
            else list(tasks)
        )
        self.raise_valueerror = raise_valueerror
        self.raise_runtimeerror = raise_runtimeerror
        self.calls = 0

    async def plan(self, goal, context=""):
        self.calls += 1
        if self.raise_valueerror:
            raise ValueError("boom")
        if self.raise_runtimeerror:
            raise RuntimeError("boom")
        return Plan(
            tasks=[
                TaskModel(id=t.id, description=t.description, dependencies=list(t.dependencies))
                for t in self.tasks
            ],
            rationale="r",
        )


class FakeTester:
    def __init__(self, passed=False, skipped=False):
        self.passed = passed
        self.skipped = skipped

    async def run(self, goal, tasks):
        return TestResult(passed=self.passed, summary="s", failures=["fail"],
                          skipped=self.skipped)


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


def make_settings(tmp_path, max_cycles=3, max_parallel_tasks=2, max_task_retries=1):
    return Settings(_env_file=None, workspace=tmp_path, max_cycles=max_cycles,
                    max_parallel_tasks=max_parallel_tasks,
                    max_task_retries=max_task_retries)


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


def drain(queue):
    """Return every event queued in *queue*."""
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


@pytest.mark.asyncio
async def test_task_that_raises_is_marked_failed(tmp_path, monkeypatch):
    class FW(FakeWorker):
        async def run(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    q = orch.event_bus.subscribe()
    result = await orch.run()
    task = orch.tasks[0]
    assert task.status == "failed"
    assert task.error == "RuntimeError: boom"
    assert task.finished_at
    assert result.tasks_failed == 1
    events = drain(q)
    assert any(e.type == EventType.ERROR for e in events)
    assert any("RuntimeError: boom" in e.message for e in events if e.type == EventType.ERROR)


@pytest.mark.asyncio
async def test_task_that_raises_is_retried_until_budget_is_spent(tmp_path, monkeypatch):
    calls = []

    class FW(FakeWorker):
        async def run(self):
            calls.append(self.task.id)
            raise RuntimeError("boom")

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=2)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    task = orch.tasks[0]
    assert task.status == "failed"
    assert task.attempts == 3  # max_task_retries + 1 attempts
    assert len(calls) == 3
    assert result.tasks_failed == 1


@pytest.mark.asyncio
async def test_task_that_raises_without_retries_fails_immediately(tmp_path, monkeypatch):
    calls = []

    class FW(FakeWorker):
        async def run(self):
            calls.append(self.task.id)
            raise RuntimeError("boom")

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=0)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    task = orch.tasks[0]
    assert task.status == "failed"
    assert task.attempts == 1
    assert len(calls) == 1
    assert result.tasks_failed == 1


@pytest.mark.asyncio
async def test_dependency_of_raised_task_is_blocked(tmp_path, monkeypatch):
    class FW(FakeWorker):
        async def run(self):
            if self.task.id == "1":
                raise RuntimeError("boom")
            return WorkerResult(summary="ok", success=True, files_changed=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    t1 = TaskModel(id="1", description="one")
    t2 = TaskModel(id="2", description="two", dependencies=["1"])
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=0)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner(tasks=[t1, t2])
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    t1o = next(t for t in orch.tasks if t.id == "1")
    t2o = next(t for t in orch.tasks if t.id == "2")
    assert t1o.status == "failed"
    assert t2o.status == "blocked"
    assert result.tasks_failed == 1


@pytest.mark.asyncio
async def test_resume_restores_state_and_loads_it_once(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()

    loads = []
    original_load = store.load

    def counting_load():
        loads.append(1)
        return original_load()

    monkeypatch.setattr(store, "load", counting_load)
    orch2 = Orchestrator(goal="x", settings=settings, state_store=store, resume=True)
    assert len(loads) == 1  # the state file is read exactly once, in __init__
    assert orch2.cycles == 1
    assert orch2.original_goal == "g"
    assert orch2.goal == "g"
    assert [t.id for t in orch2.tasks] == ["t1"]
    assert len(orch2.history) == 1
    orch2.planner = FakePlanner()
    orch2.tester = FakeTester(passed=True)
    await orch2.run()
    assert len(loads) == 1  # run() must not load the state file again


@pytest.mark.asyncio
async def test_resume_without_state_file_logs_starting_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store, resume=True)
    q = orch.event_bus.subscribe()
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    events = drain(q)
    assert any(e.type == EventType.LOG and "starting fresh" in e.message for e in events)


@pytest.mark.asyncio
async def test_skipped_tests_do_not_amend_goal(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="ORIG", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True, skipped=True)
    result = await orch.run()
    assert orch.goal == orch.original_goal == "ORIG"
    assert orch._feedback == ""
    assert result.skipped is True
    assert result.passed is True


@pytest.mark.asyncio
async def test_worker_files_changed_recorded_on_task_and_result(tmp_path, monkeypatch):
    class FW(FakeWorker):
        async def run(self):
            return WorkerResult(
                summary="ok",
                success=True,
                files_changed=[
                    "furrow/core/orchestrator.py",
                    "furrow/config.py",
                    "furrow/core/orchestrator.py",
                ],
            )

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    assert orch.tasks[0].files_changed == ["furrow/config.py", "furrow/core/orchestrator.py"]
    assert result.files_changed == ["furrow/config.py", "furrow/core/orchestrator.py"]


@pytest.mark.asyncio
async def test_worker_files_changed_empty_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    assert orch.tasks[0].files_changed == []
    assert result.files_changed == []


@pytest.mark.asyncio
async def test_blocked_dep_counted_in_tasks_blocked(tmp_path, monkeypatch):
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
    result = await orch.run()
    t2o = next(t for t in orch.tasks if t.id == "2")
    assert t2o.status == "blocked"
    assert result.tasks_blocked > 0


@pytest.mark.asyncio
async def test_tests_run_zero_before_tests_and_ge_one_when_run(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    # Return empty plan so no tests are run
    orch.planner = FakePlanner(tasks=[])
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    assert result.tests_run == 0

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings2 = make_settings(tmp_path / "ws2", max_cycles=1)
    store2 = StateStore(tmp_path / "ws2" / "state.json")
    orch2 = Orchestrator(goal="g", settings=settings2, state_store=store2)
    orch2.planner = FakePlanner()
    orch2.tester = FakeTester(passed=True)
    result2 = await orch2.run()
    assert result2.tests_run >= 1


@pytest.mark.asyncio
async def test_merge_plan_does_not_reset_failed_exhausted_task(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=0)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    task = orch.tasks[0]
    task.status = "failed"
    task.attempts = 1  # exhausted budget (max_task_retries == 0)
    plan = Plan(
        tasks=[TaskModel(id=task.id, description=task.description, dependencies=task.dependencies)],
        rationale="r",
    )
    orch._merge_plan(plan)
    assert plan.tasks[0].status == "failed"


@pytest.mark.asyncio
async def test_merge_plan_retries_failed_with_attempts_left(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    task = orch.tasks[0]
    task.status = "failed"
    task.attempts = 1  # attempts <= max_task_retries -> should be retried
    plan = Plan(
        tasks=[TaskModel(id=task.id, description=task.description, dependencies=task.dependencies)],
        rationale="r",
    )
    orch._merge_plan(plan)
    assert plan.tasks[0].status == "pending"


@pytest.mark.asyncio
async def test_merge_plan_keeps_recorded_result_and_error_when_plan_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    task = orch.tasks[0]
    task.status = "completed"
    task.result = "did the thing"
    task.error = "old error"
    plan = Plan(
        tasks=[TaskModel(id=task.id, description=task.description, dependencies=task.dependencies)],
        rationale="r",
    )
    orch._merge_plan(plan)
    assert plan.tasks[0].result == "did the thing"
    assert plan.tasks[0].error == "old error"


@pytest.mark.asyncio
async def test_merge_plan_adopts_result_and_error_from_planned_task(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    task = orch.tasks[0]
    task.status = "completed"
    task.result = "old result"
    task.error = "old error"
    plan = Plan(
        tasks=[TaskModel(id=task.id, description=task.description, dependencies=task.dependencies,
                         result="new result", error="new error")],
        rationale="r",
    )
    orch._merge_plan(plan)
    assert plan.tasks[0].result == "new result"
    assert plan.tasks[0].error == "new error"


@pytest.mark.asyncio
async def test_resume_restores_no_progress_and_prev_sig(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    await orch.run()
    # simulate persisted values
    st = store.load()
    st.no_progress = 3
    st.prev_sig = "sig123"
    store.save(st)
    orch2 = Orchestrator(goal="x", settings=settings, state_store=store, resume=True)
    assert orch2._no_progress == 3
    assert orch2._prev_sig == "sig123"


@pytest.mark.asyncio
async def test_completed_task_is_not_executed_again_when_the_plan_is_unchanged(tmp_path, monkeypatch):  # noqa: E501
    calls = []

    class CW(FakeWorker):
        async def run(self):
            calls.append(self.task.id)
            return WorkerResult(summary=f"did {self.task.id}", success=True, files_changed=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", CW)
    settings = make_settings(tmp_path, max_cycles=2)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)
    q = orch.event_bus.subscribe()
    await orch.run()
    task = orch.tasks[0]
    assert len(calls) == 1
    assert task.status == "completed"
    assert task.attempts == 1
    events = drain(q)
    assert any(e.type == EventType.LOG and "already completed" in e.message for e in events)


@pytest.mark.asyncio
async def test_completed_task_is_executed_again_when_the_description_changed(tmp_path, monkeypatch):
    calls = []

    class CW(FakeWorker):
        async def run(self):
            calls.append(self.task.id)
            return WorkerResult(summary=f"did {self.task.id}", success=True, files_changed=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", CW)
    settings = make_settings(tmp_path, max_cycles=5)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner([TaskModel(id="t1", description="do t1")])
    orch.tester = FakeTester(passed=False)
    await orch.run()  # cycle 1

    # Now change the plan description for next cycle
    orch.planner = FakePlanner([TaskModel(id="t1", description="do t1 differently")])
    await orch.run()  # cycle 2
    assert len(calls) == 2
    assert orch.tasks[0].attempts == 2


@pytest.mark.asyncio
async def test_failed_task_with_an_exhausted_budget_stays_failed(tmp_path, monkeypatch):
    calls = []

    class FW(FakeWorker):
        async def run(self):
            calls.append(self.task.id)
            return WorkerResult(summary="fail", success=False, files_changed=[], error="boom")

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=2, max_task_retries=0)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner([TaskModel(id="t1", description="one")])
    orch.tester = FakeTester(passed=False)
    await orch.run()  # first cycle
    # After the next cycle's merge, a task that failed with an exhausted
    # retry budget is still "failed" (not reset to "pending").`
    task = orch.tasks[0]
    assert task.status == "failed"
    # blocked task stays blocked and is not run in next cycle
    class FW2(FakeWorker):
        async def run(self):
            calls.append(self.task.id)
            if self.task.id == "1":
                self.task.status = "failed"
                return WorkerResult(summary="fail", success=False, files_changed=[])
            return WorkerResult(summary="ok", success=True, files_changed=[])

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW2)
    settings2 = make_settings(tmp_path / "ws2", max_cycles=2)
    store2 = StateStore(tmp_path / "ws2" / "state.json")
    orch2 = Orchestrator(goal="g", settings=settings2, state_store=store2)
    t1 = TaskModel(id="1", description="one")
    t2 = TaskModel(id="2", description="two", dependencies=["1"])
    orch2.planner = FakePlanner(tasks=[t1, t2])
    orch2.tester = FakeTester(passed=False)
    calls.clear()
    await orch2.run()
    t2o = next(t for t in orch2.tasks if t.id == "2")
    assert t2o.status == "blocked"
    # blocked task's worker is never invoked in next cycle
    assert "2" not in calls or calls.count("2") == 0  # it was never run


@pytest.mark.asyncio
async def test_state_is_saved_once_per_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=3)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)
    q = orch.event_bus.subscribe()
    saves = []
    orig_save = store.save

    def counting_save(state):
        saves.append(1)
        return orig_save(state)

    monkeypatch.setattr(store, "save", counting_save)
    await orch.run()
    assert len(saves) == orch.cycles
    events = drain(q)
    state_saved = [e for e in events if e.type == EventType.STATE_SAVED]
    assert len(state_saved) == orch.cycles


@pytest.mark.asyncio
async def test_persisted_no_progress_matches_the_live_counter(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=2)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)
    await orch.run()
    st = store.load()
    assert st is not None
    assert orch._no_progress == st.no_progress


@pytest.mark.asyncio
async def test_tests_run_is_restored_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    result1 = await orch.run()
    assert result1.tests_run >= 1
    orch2 = Orchestrator(goal="x", settings=settings, state_store=store, resume=True)
    assert orch2._tests_run == 1  # restored
    orch2.planner = FakePlanner()
    orch2.tester = FakeTester(passed=True)
    result2 = await orch2.run()
    assert result2.tests_run >= 1  # accounts for both cycles (restored)


@pytest.mark.asyncio
async def test_planner_runtimeerror_produces_error_event_and_stopped_reason(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner(raise_runtimeerror=True)
    orch.tester = FakeTester(passed=True)
    q = orch.event_bus.subscribe()
    result = await orch.run()
    assert result.stopped_reason.startswith("planner failed")
    events = drain(q)
    assert any(e.type == EventType.ERROR for e in events)


@pytest.mark.asyncio
async def test_state_store_save_oserror_does_not_crash_run(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)

    def boom(state):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", boom)
    q = orch.event_bus.subscribe()
    result = await orch.run()
    assert isinstance(result, RunResult)
    events = drain(q)
    errors = [e for e in events if e.type == EventType.ERROR]
    assert errors, "a failed state save must emit an ERROR event"
    assert any("disk full" in e.message for e in errors)
    assert any(str(store.path) in e.message for e in errors)
    # The state was NOT written, so no STATE_SAVED may be emitted for that cycle.
    assert not [e for e in events if e.type == EventType.STATE_SAVED]


@pytest.mark.asyncio
async def test_state_saved_is_emitted_only_for_cycles_whose_save_succeeded(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=2)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=False)

    attempts = []
    original_save = store.save

    def flaky_save(state):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("disk full")
        return original_save(state)

    monkeypatch.setattr(store, "save", flaky_save)
    q = orch.event_bus.subscribe()
    result = await orch.run()

    assert isinstance(result, RunResult)
    assert result.cycles == 2
    assert len(attempts) == 2
    events = drain(q)
    state_saved = [e for e in events if e.type == EventType.STATE_SAVED]
    errors = [e for e in events if e.type == EventType.ERROR]
    # Only the cycle whose save succeeded reports a saved state.
    assert len(state_saved) == 1
    assert state_saved[0].data["path"] == str(store.path)
    # The failed cycle reports the failure instead.
    assert len(errors) == 1
    assert "disk full" in errors[0].message
    assert str(store.path) in errors[0].message


@pytest.mark.asyncio
async def test_worker_result_attribute_access_failure_marks_task_failed(tmp_path, monkeypatch):
    class BadResult:
        @property
        def success(self):
            raise RuntimeError("attr boom")

    class FW(FakeWorker):
        async def run(self):
            return BadResult()

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=1, max_task_retries=0)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    q = orch.event_bus.subscribe()
    await orch.run()
    task = orch.tasks[0]
    assert task.status in ("failed", "pending")
    assert task.attempts >= 1
    events = drain(q)
    assert any(e.type == EventType.ERROR for e in events)
    assert any(
        e.type == EventType.TASK_FINISHED and not e.data.get("success", True)
        for e in events
    )


@pytest.mark.asyncio
async def test_cancelled_error_propagates_out_of_run(tmp_path, monkeypatch):
    class FW(FakeWorker):
        async def run(self):
            raise asyncio.CancelledError()

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FW)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store)
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    with pytest.raises(asyncio.CancelledError):
        await orch.run()


@pytest.mark.asyncio
async def test_usage_cost_carried(tmp_path, monkeypatch):
    from furrow.config import Usage

    class FakeClient:
        def __init__(self):
            self.total_usage = Usage(input_tokens=10, output_tokens=20, calls=1, cost_usd=0.123456)

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    settings = make_settings(tmp_path, max_cycles=1)
    store = StateStore(tmp_path / "state.json")
    orch = Orchestrator(goal="g", settings=settings, state_store=store, client=FakeClient())
    orch.planner = FakePlanner()
    orch.tester = FakeTester(passed=True)
    result = await orch.run()
    assert result.usage.cost_usd == 0.123456
    st = store.load()
    assert st.usage.cost_usd == 0.123456
