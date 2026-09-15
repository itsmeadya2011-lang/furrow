import asyncio
import io

import pytest

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    def __init__(self, plans):
        self.plans = list(plans)
        self.goals = []

    async def plan(self, goal):
        self.goals.append(goal)
        if self.plans:
            return self.plans.pop(0)
        return Plan(tasks=[], rationale="")


class FakeWorker:
    active = 0
    max_active = 0
    instances = []

    def __init__(self, task, client=None):
        self.task = task
        self.client = client
        type(self).instances.append(self)

    async def run(self):
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)
        await asyncio.sleep(0)
        type(self).active -= 1
        if self.task.id == "fail":
            raise RuntimeError("worker failed")
        return f"completed {self.task.id}"


class FakeTester:
    instances = []

    def __init__(self, client=None, **kwargs):
        self.client = client
        self.result = TestResult(passed=True, summary="ok", failures=[])
        type(self).instances.append(self)

    async def run(self, goal, tasks):
        return self.result


@pytest.fixture
def settings(tmp_path):
    return Settings(
        max_cycles=0,
        max_parallel_tasks=5,
        workspace=tmp_path,
    )


@pytest.fixture
def fake_client(settings):
    class Client:
        def __init__(self):
            self.settings = settings

    return Client()


def install_fakes(monkeypatch, planner, tester_result=None):
    monkeypatch.setattr(
        orchestrator_module,
        "PlannerAgent",
        lambda **kwargs: planner,
    )
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    FakeWorker.active = 0
    FakeWorker.max_active = 0
    FakeWorker.instances = []
    if tester_result is not None:
        class ResultTester(FakeTester):
            async def run(self, goal, tasks):
                return tester_result

        monkeypatch.setattr(orchestrator_module, "TesterAgent", ResultTester)
    else:
        monkeypatch.setattr(orchestrator_module, "TesterAgent", FakeTester)


def test_get_tasks_selects_dependency_ready_pending_tasks(settings, fake_client):
    dependency = TaskModel(id="dependency", description="dependency")
    child = TaskModel(
        id="child",
        description="child",
        dependencies=["dependency"],
    )
    independent = TaskModel(id="independent", description="independent")
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)
    orchestrator.tasks = [dependency, child, independent]
    orchestrator._tasks_by_id = {task.id: task for task in orchestrator.tasks}

    assert [task.id for task in orchestrator._get_tasks()] == [
        "dependency",
        "independent",
    ]


def test_get_tasks_respects_parallel_limit(settings, fake_client):
    tasks = [TaskModel(id=str(index), description="task") for index in range(4)]
    settings.max_parallel_tasks = 2
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)
    orchestrator.tasks = tasks
    orchestrator._tasks_by_id = {task.id: task for task in tasks}

    assert [task.id for task in orchestrator._get_tasks()] == ["0", "1"]


async def test_dependency_state_is_preserved_across_cycles(
    monkeypatch, settings, fake_client
):
    root = TaskModel(id="root", description="root")
    child = TaskModel(id="child", description="child", dependencies=["root"])
    planner = FakePlanner([Plan(tasks=[root, child], rationale="plan")])
    install_fakes(monkeypatch, planner)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)

    await orchestrator.run()

    assert [worker.task.id for worker in FakeWorker.instances] == [
        "root",
        "child",
    ]
    assert {task.id: task.status for task in orchestrator.tasks} == {
        "root": "completed",
        "child": "completed",
    }
    assert orchestrator.plan is not None


async def test_blocked_task_is_reconciled_when_dependency_arrives(
    monkeypatch, settings, fake_client
):
    child = TaskModel(
        id="child",
        description="child",
        dependencies=["dependency"],
    )
    dependency = TaskModel(id="dependency", description="dependency")
    planner = FakePlanner(
        [
            Plan(tasks=[child], rationale="plan with missing dependency"),
            Plan(tasks=[dependency, child], rationale="resolved plan"),
        ]
    )
    install_fakes(monkeypatch, planner)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)

    await orchestrator.run()

    assert [worker.task.id for worker in FakeWorker.instances] == [
        "dependency",
        "child",
    ]
    child_task = next(task for task in orchestrator.tasks if task.id == "child")
    assert child_task.status == "completed"


async def test_parallel_limit_is_respected(monkeypatch, settings, fake_client):
    tasks = [TaskModel(id=str(index), description="task") for index in range(5)]
    settings.max_parallel_tasks = 2
    planner = FakePlanner([Plan(tasks=tasks, rationale="plan")])
    install_fakes(monkeypatch, planner)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)

    await orchestrator.run()

    assert {worker.task.id for worker in FakeWorker.instances} == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }
    assert FakeWorker.max_active == 2


async def test_max_cycles_stops_the_run(monkeypatch, settings, fake_client):
    settings.max_cycles = 2
    root = TaskModel(id="root", description="root")
    child = TaskModel(id="child", description="child", dependencies=["root"])
    planner = FakePlanner([Plan(tasks=[root, child], rationale="plan")])
    install_fakes(monkeypatch, planner)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert planner.goals == ["goal", "goal"]
    assert {task.id: task.status for task in orchestrator.tasks} == {
        "root": "completed",
        "child": "completed",
    }


async def test_failed_dependency_blocks_dependent(
    monkeypatch,
    settings,
    fake_client,
):
    root = TaskModel(id="fail", description="root")
    child = TaskModel(id="child", description="child", dependencies=["fail"])
    planner = FakePlanner([Plan(tasks=[root, child], rationale="plan")])
    install_fakes(monkeypatch, planner)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)
    output = io.StringIO()
    monkeypatch.setattr(
        orchestrator_module,
        "console",
        orchestrator_module.Console(file=output, force_terminal=False),
    )

    await orchestrator.run()

    root_task = next(task for task in orchestrator.tasks if task.id == "fail")
    child_task = next(task for task in orchestrator.tasks if task.id == "child")
    assert root_task.status == "failed"
    assert child_task.status == "blocked"
    assert "fail" in child_task.result

    rendered = output.getvalue()
    assert "Run complete. Halting." not in rendered
    assert "Task failure detected. Halting." in rendered


async def test_repeated_unresolved_dependency_plan_halts_without_progress(
    monkeypatch, settings, fake_client
):
    child = TaskModel(
        id="child",
        description="child",
        dependencies=["missing-dependency"],
    )
    plan = Plan(tasks=[child], rationale="plan with missing dependency")
    planner = FakePlanner([plan, plan])
    install_fakes(monkeypatch, planner)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)
    output = io.StringIO()
    monkeypatch.setattr(
        orchestrator_module,
        "console",
        orchestrator_module.Console(file=output, force_terminal=False),
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert planner.goals == ["goal", "goal"]
    rendered = output.getvalue()
    assert "No progress made. Halting." in rendered


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", True),
        ("failed", True),
        ("blocked", False),
    ],
)
def test_terminal_task_states_stop_progress(settings, fake_client, status, expected):
    task = TaskModel(id="task", description="task", status=status)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)
    orchestrator.tasks = [task]
    orchestrator._tasks_by_id = {task.id: task}

    assert orchestrator._is_done() is expected


def test_cyclic_pending_tasks_stop_progress(settings, fake_client):
    root = TaskModel(id="root", description="root", dependencies=["child"])
    child = TaskModel(id="child", description="child", dependencies=["root"])
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)
    orchestrator.tasks = [root, child]
    orchestrator._tasks_by_id = {task.id: task for task in orchestrator.tasks}

    assert orchestrator._is_done() is True


async def test_cyclic_pending_tasks_halt_as_stalled(
    monkeypatch, settings, fake_client
):
    root = TaskModel(id="root", description="root", dependencies=["child"])
    child = TaskModel(id="child", description="child", dependencies=["root"])
    planner = FakePlanner([Plan(tasks=[root, child], rationale="cyclic plan")])
    install_fakes(monkeypatch, planner)
    orchestrator = Orchestrator("goal", client=fake_client, settings=settings)
    output = io.StringIO()
    monkeypatch.setattr(
        orchestrator_module,
        "console",
        orchestrator_module.Console(file=output, force_terminal=False),
    )

    await orchestrator.run()

    assert orchestrator.cycles == 1
    rendered = output.getvalue()
    assert "Work stalled. Halting." in rendered
    assert "Run complete. Halting." not in rendered


async def test_test_failures_do_not_replace_goal(
    monkeypatch, settings, fake_client
):
    root = TaskModel(id="root", description="root")
    child = TaskModel(id="child", description="child", dependencies=["root"])
    planner = FakePlanner([Plan(tasks=[root, child], rationale="plan")])
    failed_test = TestResult(
        passed=False,
        summary="tests failed",
        failures=["assertion failed"],
    )
    install_fakes(monkeypatch, planner, tester_result=failed_test)
    orchestrator = Orchestrator(
        "original goal",
        client=fake_client,
        settings=settings,
    )
    output = io.StringIO()
    monkeypatch.setattr(
        orchestrator_module,
        "console",
        orchestrator_module.Console(file=output, force_terminal=False),
    )

    await orchestrator.run()

    assert orchestrator.goal == "original goal"
    assert orchestrator.original_goal == "original goal"
    assert planner.goals == ["original goal", "original goal"]
    assert orchestrator.cycles == 2
    assert orchestrator.last_test_result is failed_test

    rendered = output.getvalue()
    assert "Tests failed. Halting." in rendered
    assert "Run complete. Halting." not in rendered
