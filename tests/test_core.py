from functools import partial

import pytest

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class _FakePlanner:
    def __init__(self, client, plans, goals):
        self.client = client
        self._plans = plans
        self.goals = goals

    async def plan(self, goal):
        self.goals.append(goal)
        return next(self._plans)


class _FakeWorker:
    def __init__(self, task, client, outcomes, calls):
        self.task = task
        self.client = client
        self._outcomes = outcomes
        self.calls = calls

    async def run(self):
        self.calls.append(self.task)
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeTester:
    def __init__(self, client, results, calls):
        self.client = client
        self._results = results
        self.calls = calls

    async def run(self, goal, tasks):
        self.calls.append((goal, tasks))
        return next(self._results)


def _install_fakes(monkeypatch, plans, test_results, worker_outcomes=()):
    planner_goals = []
    tester_calls = []
    worker_calls = []

    monkeypatch.setattr(
        orchestrator_module,
        "PlannerAgent",
        partial(_FakePlanner, plans=iter(plans), goals=planner_goals),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "WorkerAgent",
        partial(_FakeWorker, outcomes=iter(worker_outcomes), calls=worker_calls),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "TesterAgent",
        partial(_FakeTester, results=iter(test_results), calls=tester_calls),
    )
    return planner_goals, tester_calls, worker_calls


def _settings(max_cycles=0):
    return Settings(max_cycles=max_cycles)


def _task(description="task", status="pending"):
    return TaskModel(id="1", description=description, status=status)


def _plan(*tasks, rationale="plan"):
    return Plan(tasks=list(tasks), rationale=rationale)


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_orchestrator_completes_after_completed_tasks_and_passing_tests(monkeypatch):
    goal = "implement the feature"
    task = _task()
    plan = _plan(task)
    planner_goals, tester_calls, worker_calls = _install_fakes(
        monkeypatch,
        [plan],
        [TestResult(passed=True, summary="all tests passed", failures=[])],
        ["worker result"],
    )

    orchestrator = Orchestrator(goal=goal, settings=_settings())
    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator._is_done() is True
    assert task.status == "completed"
    assert task.result == "worker result"
    assert planner_goals == [goal]
    assert worker_calls == [task]
    assert tester_calls == [(goal, plan.tasks)]


def test_orchestrator_does_not_complete_with_incomplete_task():
    goal = "implement the feature"
    task = _task(status="in_progress")
    orchestrator = Orchestrator(goal=goal, settings=_settings())
    orchestrator.current_plan = _plan(task)
    orchestrator.latest_test_result = TestResult(
        passed=True, summary="all tests passed", failures=[]
    )

    assert orchestrator._is_done() is False
    assert orchestrator.goal == goal
    assert orchestrator.original_goal == goal


@pytest.mark.asyncio
async def test_orchestrator_retries_after_worker_failure(monkeypatch):
    goal = "implement the feature"
    first_task = _task("first task")
    second_task = _task("second task")
    planner_goals, _, worker_calls = _install_fakes(
        monkeypatch,
        [_plan(first_task), _plan(second_task)],
        [
            TestResult(passed=True, summary="tests passed", failures=[]),
            TestResult(passed=True, summary="tests passed after retry", failures=[]),
        ],
        [RuntimeError("worker failed"), "fixed"],
    )

    orchestrator = Orchestrator(goal=goal, settings=_settings())
    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator._is_done() is True
    assert first_task.status == "failed"
    assert first_task.result == "worker failed"
    assert second_task.status == "completed"
    expected_retry_goal = (
        f"{goal}\n\nWorker failures:\nTask {first_task.id} failed: worker failed"
    )
    assert planner_goals == [goal, expected_retry_goal]
    assert orchestrator.goal == goal
    assert orchestrator.original_goal == goal
    assert worker_calls == [first_task, second_task]


@pytest.mark.asyncio
async def test_failed_tests_trigger_planning_with_original_goal(monkeypatch):
    goal = "implement the feature"
    failure = "test_api.py::test_feature failed"
    first_task = _task("first task")
    second_task = _task("second task")
    planner_goals, _, _ = _install_fakes(
        monkeypatch,
        [_plan(first_task), _plan(second_task)],
        [
            TestResult(passed=False, summary="one failure", failures=[failure]),
            TestResult(passed=True, summary="all tests passed", failures=[]),
        ],
        ["first result", "second result"],
    )

    orchestrator = Orchestrator(goal=goal, settings=_settings())
    await orchestrator.run()

    expected_retry_goal = f"{goal}\n\nLatest test failures:\n{failure}"
    assert orchestrator.cycles == 2
    assert planner_goals == [goal, expected_retry_goal]
    assert orchestrator.goal == goal
    assert orchestrator.original_goal == goal
    assert orchestrator.latest_test_result.passed is True


@pytest.mark.asyncio
async def test_max_cycles_stops_unfinished_run(monkeypatch):
    goal = "implement the feature"
    first_task = _task("first task")
    second_task = _task("second task")
    planner_goals, _, _ = _install_fakes(
        monkeypatch,
        [_plan(first_task), _plan(second_task)],
        [
            TestResult(passed=True, summary="tests passed", failures=[]),
            TestResult(passed=True, summary="tests passed", failures=[]),
        ],
        [RuntimeError("worker failed"), RuntimeError("worker still failed")],
    )

    orchestrator = Orchestrator(goal=goal, settings=_settings(max_cycles=2))
    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert len(planner_goals) == 2
    assert orchestrator._is_done() is False
    assert second_task.status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(("first_test_passes", "expected_cycles"), [(True, 1), (False, 2)])
async def test_empty_plan_is_tested_and_has_predictable_completion(
    monkeypatch, first_test_passes, expected_cycles
):
    goal = "verify the feature"
    failure = "empty plan did not satisfy the goal"
    planner_goals, tester_calls, worker_calls = _install_fakes(
        monkeypatch,
        [_plan(), _plan()],
        [
            TestResult(
                passed=first_test_passes,
                summary="empty plan result",
                failures=[] if first_test_passes else [failure],
            ),
            TestResult(passed=True, summary="all tests passed", failures=[]),
        ],
    )

    orchestrator = Orchestrator(goal=goal, settings=_settings(max_cycles=2))
    await orchestrator.run()

    assert orchestrator.cycles == expected_cycles
    assert orchestrator.current_plan.tasks == []
    assert worker_calls == []
    assert [tasks for _, tasks in tester_calls] == [[] for _ in tester_calls]
    if first_test_passes:
        assert planner_goals == [goal]
        assert tester_calls == [(goal, [])]
    else:
        expected_retry_goal = f"{goal}\n\nLatest test failures:\n{failure}"
        assert planner_goals == [goal, expected_retry_goal]
        assert tester_calls == [(goal, []), (goal, [])]
    assert orchestrator._is_done() is True
