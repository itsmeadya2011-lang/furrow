import asyncio
import json

import pytest

from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.orchestrator import Orchestrator


class SequencePlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        return self.plans.pop(0)


class RepeatedPlanner:
    def __init__(self, plan: Plan) -> None:
        self.value = plan
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        return self.value


class RecordingWorker:
    def __init__(self) -> None:
        self.ids: list[str] = []

    async def __call__(self, task: TaskModel) -> str:
        self.ids.append(task.id)
        return f"result:{task.id}"


class ConcurrencyWorker:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.ids: list[str] = []

    async def __call__(self, task: TaskModel) -> str:
        self.ids.append(task.id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return task.id


class FailingWorker:
    async def __call__(self, task: TaskModel) -> str:
        if task.id == "first":
            raise RuntimeError("worker failure")
        return task.id


class PassingTester:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[TaskModel]]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls.append((goal, list(tasks)))
        return TestResult(passed=True, summary="ok", failures=[])


class FailingTester:
    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return TestResult(passed=False, summary="failure", failures=["test failed"])


class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return json.dumps({"tasks": [], "rationale": "done"})


def dependency_plan() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="first", description="first"),
            TaskModel(id="second", description="second", dependencies=["first"]),
        ],
        rationale="ordered",
    )


@pytest.mark.asyncio
async def test_retains_tasks_across_cycles_and_terminates() -> None:
    planner = SequencePlanner(
        [
            dependency_plan(),
            Plan(
                tasks=[
                    TaskModel(
                        id="second",
                        description="second",
                        dependencies=["first"],
                    )
                ],
                rationale="remaining",
            ),
        ]
    )
    worker = RecordingWorker()
    tester = PassingTester()
    orchestrator = Orchestrator(
        "original goal",
        planner=planner,
        worker_factory=worker,
        tester=tester,
        settings=Settings(max_cycles=3),
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator.goal == "original goal"
    assert planner.goals == ["original goal", "original goal"]
    assert worker.ids == ["first", "second"]
    assert [task.status for task in orchestrator.tasks] == [
        TaskStatus.COMPLETED,
        TaskStatus.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_dependency_is_not_run_before_completed_dependency() -> None:
    planner = RepeatedPlanner(dependency_plan())
    worker = RecordingWorker()
    tester = PassingTester()
    orchestrator = Orchestrator(
        "original goal",
        planner=planner,
        worker_factory=worker,
        tester=tester,
        settings=Settings(max_cycles=3),
    )

    await orchestrator.run()

    assert worker.ids == ["first", "second"]
    assert orchestrator.tasks[1].result == "result:second"


@pytest.mark.asyncio
async def test_dependency_cycle_is_rejected_without_unbounded_loop() -> None:
    plan = Plan(
        tasks=[
            TaskModel(
                id="first",
                description="first",
                dependencies=["second"],
            ),
            TaskModel(
                id="second",
                description="second",
                dependencies=["first"],
            ),
        ],
        rationale="cyclic",
    )
    planner = RepeatedPlanner(plan)
    worker = RecordingWorker()
    tester = PassingTester()
    orchestrator = Orchestrator(
        "original goal",
        planner=planner,
        worker_factory=worker,
        tester=tester,
        settings=Settings(max_cycles=0),
    )

    with pytest.raises(ValueError, match="dependency cycle detected"):
        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert planner.goals == ["original goal"]
    assert worker.ids == []


@pytest.mark.asyncio
async def test_concurrency_is_bounded() -> None:
    plan = Plan(
        tasks=[TaskModel(id=f"task-{index}", description=f"task {index}") for index in range(5)],
        rationale="parallel",
    )
    worker = ConcurrencyWorker()
    tester = PassingTester()
    orchestrator = Orchestrator(
        "original goal",
        planner=RepeatedPlanner(plan),
        worker_factory=worker,
        tester=tester,
        settings=Settings(max_cycles=1, max_parallel_tasks=2),
    )

    await orchestrator.run()

    assert worker.max_active == 2
    assert len(worker.ids) == 5
    assert all(task.status == TaskStatus.COMPLETED for task in orchestrator.tasks)


@pytest.mark.asyncio
async def test_max_cycles_stops_the_loop() -> None:
    plan = Plan(tasks=[TaskModel(id="task", description="task")], rationale="one")
    planner = RepeatedPlanner(plan)
    worker = RecordingWorker()
    tester = FailingTester()
    orchestrator = Orchestrator(
        "original goal",
        planner=planner,
        worker_factory=worker,
        tester=tester,
        settings=Settings(max_cycles=1),
    )

    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert planner.goals == ["original goal"]
    assert worker.ids == ["task"]


@pytest.mark.asyncio
async def test_failed_dependency_fails_dependent_task() -> None:
    plan = dependency_plan()
    worker = FailingWorker()
    tester = PassingTester()
    orchestrator = Orchestrator(
        "original goal",
        planner=RepeatedPlanner(plan),
        worker_factory=worker,
        tester=tester,
        settings=Settings(max_cycles=2),
    )

    await orchestrator.run()

    first, second = orchestrator.tasks
    assert orchestrator.cycles == 1
    assert planner.goals == ["original goal"]
    assert worker.ids == ["first"]
    assert first.status == TaskStatus.FAILED
    assert first.result == "worker failure"
    assert second.status == TaskStatus.FAILED
    assert "dependency failed" in (second.result or "")
    assert "first" in (second.result or "")


def test_task_status_supports_string_comparison_and_json() -> None:
    task = TaskModel(id="task", description="work", status="completed")

    assert task.status == "completed"
    assert task.status == TaskStatus.COMPLETED
    assert json.loads(task.model_dump_json())["status"] == "completed"

    task.status = "failed"
    assert task.status == "failed"


def test_plan_rejects_duplicate_ids_and_unknown_dependencies() -> None:
    with pytest.raises(ValueError, match="duplicate task id"):
        Plan(
            tasks=[TaskModel(id="same", description="same") for _ in range(2)],
            rationale="invalid",
        )

    plan = Plan(
        tasks=[TaskModel(id="task", description="task", dependencies=["missing"])],
        rationale="invalid",
    )
    with pytest.raises(ValueError, match="unknown dependency"):
        plan.validate_graph()


@pytest.mark.asyncio
async def test_fake_client_without_provider_keys_is_supported() -> None:
    client = FakeClient()
    tester = PassingTester()
    orchestrator = Orchestrator(
        "original goal",
        client=client,
        tester=tester,
        settings=Settings(max_cycles=1),
    )

    await orchestrator.run()

    assert len(client.prompts) == 1
    assert "original goal" in client.prompts[0]
    assert orchestrator.tasks == []
