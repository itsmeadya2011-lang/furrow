"""Loop-level tests for :class:`~furrow.core.orchestrator.Orchestrator`.

Every agent the orchestrator drives is replaced with an async fake, so these
tests exercise the real control flow (cycle counting, task bookkeeping, goal
rewriting, halt conditions) without a network, an API key, or a real test suite.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

# ``TesterAgent`` and ``TestResult`` are reached through their modules: their names
# start with "Test", so binding them here would make pytest try to collect them.
from furrow import config
from furrow.agents import tester as tester_module
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator

GOAL = "Add JWT auth to the API"

PlanFactory = Callable[[int], Plan]
WorkerFactory = Callable[[TaskModel], Awaitable[str]]
TesterFactory = Callable[[str, list[TaskModel]], Awaitable[config.TestResult]]


class FakeClient:
    """Duck-typed stand-in for :class:`~furrow.llm.LLMClient`.

    The orchestrator itself only reads ``client.settings``; the agents it builds
    are all faked. ``complete`` raises so an accidental real LLM call inside the
    loop fails loudly instead of reaching the network.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        raise AssertionError("orchestrator loop must not call the LLM directly")


def make_client(**overrides: Any) -> FakeClient:
    return FakeClient(Settings(_env_file=None, **overrides))


def make_orchestrator(client: FakeClient) -> Orchestrator:
    return Orchestrator(goal=GOAL, client=client)  # type: ignore[arg-type]


def make_plan(*descriptions: str) -> Plan:
    return Plan(
        tasks=[
            TaskModel(id=str(index), description=description, files=[f"src/mod_{index}.py"])
            for index, description in enumerate(descriptions, start=1)
        ],
        rationale="canned plan",
    )


def passed(summary: str = "all green") -> config.TestResult:
    return config.TestResult(passed=True, summary=summary, failures=[])


def failed(*failures: str, summary: str = "1 failed") -> config.TestResult:
    return config.TestResult(passed=False, summary=summary, failures=list(failures))


class FakeAgents:
    """Async fakes for the three agents the orchestrator drives, plus a call log."""

    def __init__(self, plan: PlanFactory, worker: WorkerFactory, tester: TesterFactory) -> None:
        self._plan = plan
        self._worker = worker
        self._tester = tester
        self.planner_goals: list[str] = []
        self.worker_tasks: list[TaskModel] = []
        self.tester_goals: list[str] = []
        self.tester_task_batches: list[list[TaskModel]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        log = self

        async def fake_plan(self: PlannerAgent, goal: str) -> Plan:
            log.planner_goals.append(goal)
            return log._plan(len(log.planner_goals) - 1)

        async def fake_worker_run(self: WorkerAgent) -> str:
            log.worker_tasks.append(self.task)
            return await log._worker(self.task)

        async def fake_tester_run(
            self: tester_module.TesterAgent, goal: str, tasks: list[TaskModel]
        ) -> config.TestResult:
            log.tester_goals.append(goal)
            log.tester_task_batches.append(list(tasks))
            return await log._tester(goal, tasks)

        monkeypatch.setattr(PlannerAgent, "plan", fake_plan)
        monkeypatch.setattr(WorkerAgent, "run", fake_worker_run)
        monkeypatch.setattr(tester_module.TesterAgent, "run", fake_tester_run)


def install_agents(
    monkeypatch: pytest.MonkeyPatch,
    plan: PlanFactory,
    tester: TesterFactory,
    worker: WorkerFactory | None = None,
) -> FakeAgents:
    async def default_worker(task: TaskModel) -> str:
        return f"done: {task.id}"

    agents = FakeAgents(plan=plan, worker=worker or default_worker, tester=tester)
    agents.install(monkeypatch)
    return agents


async def test_loop_runs_until_tasks_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = make_orchestrator(make_client())

    async def always_passes(goal: str, tasks: list[TaskModel]) -> config.TestResult:
        return passed()

    agents = install_agents(
        monkeypatch,
        plan=lambda cycle: make_plan("write the token helper", "wire it into the API"),
        tester=always_passes,
    )

    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert agents.planner_goals == [GOAL]
    assert [task.id for task in agents.worker_tasks] == ["1", "2"]
    assert [task.status for task in orchestrator.tasks] == ["completed", "completed"]
    assert all(task.result for task in orchestrator.tasks)
    assert len(agents.tester_task_batches) == 1
    assert [task.id for task in agents.tester_task_batches[0]] == ["1", "2"]
    assert orchestrator._is_done() is True
    assert orchestrator.goal == GOAL


async def test_max_cycles_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = make_orchestrator(make_client(max_cycles=2))

    async def always_fails(goal: str, tasks: list[TaskModel]) -> config.TestResult:
        return failed("tests/test_api.py::test_login")

    agents = install_agents(
        monkeypatch,
        plan=lambda cycle: make_plan(f"attempt {cycle}"),
        tester=always_fails,
    )

    await orchestrator.run()

    # Exactly two cycles ran, and the third was refused before doing any work.
    assert orchestrator.cycles == 2
    assert len(agents.planner_goals) == 2
    assert len(agents.worker_tasks) == 2
    assert len(agents.tester_task_batches) == 2
    assert all(task.status == "completed" for task in orchestrator.tasks)


async def test_failing_tests_preserve_original_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = make_orchestrator(make_client(max_cycles=2))

    async def always_fails(goal: str, tasks: list[TaskModel]) -> config.TestResult:
        return failed("tests/test_api.py::test_login", "tests/test_api.py::test_refresh")

    agents = install_agents(
        monkeypatch,
        plan=lambda cycle: make_plan(f"attempt {cycle}"),
        tester=always_fails,
    )

    await orchestrator.run()

    assert orchestrator.original_goal == GOAL
    assert orchestrator.goal.startswith(GOAL)
    assert "tests/test_api.py::test_login" in orchestrator.goal
    assert "tests/test_api.py::test_refresh" in orchestrator.goal
    # The rewritten goal is what the next cycle is planned and tested against.
    assert agents.planner_goals[0] == GOAL
    assert "tests/test_api.py::test_login" in agents.planner_goals[1]
    assert "tests/test_api.py::test_login" in agents.tester_goals[1]


async def test_max_cycles_zero_never_stops_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stop(Exception):
        pass

    orchestrator = make_orchestrator(make_client(max_cycles=0))
    seen: list[int] = []

    async def fails_then_stops(goal: str, tasks: list[TaskModel]) -> config.TestResult:
        seen.append(len(tasks))
        if len(seen) == 4:
            raise Stop
        return failed("tests/test_api.py::test_login")

    install_agents(
        monkeypatch,
        plan=lambda cycle: make_plan(f"attempt {cycle}"),
        tester=fails_then_stops,
    )

    with pytest.raises(Stop):
        await orchestrator.run()

    # The loop only stopped because the fake tester raised, never on a cycle cap.
    assert seen == [1, 1, 1, 1]
    assert orchestrator.cycles == 4


async def test_empty_plan_halts_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = make_orchestrator(make_client())

    async def never_called(goal: str, tasks: list[TaskModel]) -> config.TestResult:
        raise AssertionError("the tester must not run when the plan is empty")

    agents = install_agents(
        monkeypatch,
        plan=lambda cycle: Plan(tasks=[], rationale="nothing to do"),
        tester=never_called,
    )

    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator.tasks == []
    assert agents.worker_tasks == []
    assert agents.tester_goals == []
    assert orchestrator._is_done() is True
    assert orchestrator.goal == GOAL


async def test_worker_exception_fails_task_and_blocks_done(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = make_orchestrator(make_client(max_cycles=1))

    async def exploding_worker(task: TaskModel) -> str:
        raise RuntimeError(f"worker blew up on {task.id}")

    async def always_passes(goal: str, tasks: list[TaskModel]) -> config.TestResult:
        return passed()

    install_agents(
        monkeypatch,
        plan=lambda cycle: make_plan("do the thing"),
        tester=always_passes,
        worker=exploding_worker,
    )

    await orchestrator.run()

    assert [task.status for task in orchestrator.tasks] == ["failed"]
    assert "worker blew up on 1" in (orchestrator.tasks[0].result or "")
    assert orchestrator._is_done() is False


def test_is_done_requires_every_task_to_have_completed() -> None:
    orchestrator = make_orchestrator(make_client())
    orchestrator.tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]

    assert orchestrator._is_done() is False

    orchestrator.tasks[0].status = "completed"
    assert orchestrator._is_done() is False

    orchestrator.tasks[1].status = "completed"
    assert orchestrator._is_done() is True

    orchestrator.tasks[1].status = "failed"
    assert orchestrator._is_done() is False


def test_failing_test_run_blocks_done_even_when_tasks_completed() -> None:
    orchestrator = make_orchestrator(make_client())
    orchestrator.tasks = [TaskModel(id="1", description="a", status="completed")]

    assert orchestrator._is_done() is True

    orchestrator.last_test_result = failed("tests/test_api.py::test_login")
    assert orchestrator._is_done() is False

    orchestrator.last_test_result = passed()
    assert orchestrator._is_done() is True


def test_is_done_is_true_without_tasks() -> None:
    assert make_orchestrator(make_client())._is_done() is True
