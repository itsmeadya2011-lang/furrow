import asyncio
from types import SimpleNamespace

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeClient:
    def __init__(self) -> None:
        self.settings = Settings()
        self.active = 0
        self.max_active = 0

    async def complete(self, prompt: str, model: str | None = None) -> str:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return "done"


class FakePlanner:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.calls: list[dict[str, object]] = []

    async def plan(self, goal: str, failures: list[str] | None = None) -> Plan:
        self.calls.append({"goal": goal, "failures": list(failures or [])})
        return self.plan


class FakeTester:
    def __init__(self, result: TestResult) -> None:
        self.result = result
        self.calls: list[tuple[str, list[TaskModel]]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls.append((goal, tasks))
        return self.result


def make_plan() -> Plan:
    return Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")


@pytest.mark.asyncio
async def test_orchestrator_rejects_negative_max_cycles():
    with pytest.raises(ValueError, match="^max_cycles must be >= 0$"):
        Orchestrator(
            "goal",
            client=FakeClient(),
            planner=FakePlanner(make_plan()),
            tester=FakeTester(
                TestResult(passed=True, summary="ok", failures=[])
            ),
            max_cycles=-1,
        )


@pytest.mark.asyncio
async def test_orchestrator_rejects_max_parallel_tasks_below_one():
    with pytest.raises(ValueError, match="^max_parallel_tasks must be >= 1$"):
        Orchestrator(
            "goal",
            client=FakeClient(),
            planner=FakePlanner(make_plan()),
            tester=FakeTester(
                TestResult(passed=True, summary="ok", failures=[])
            ),
            max_parallel_tasks=0,
        )


@pytest.mark.asyncio
async def test_orchestrator_tracks_completion():
    task = TaskModel(id="1", description="do thing")
    plan = Plan(tasks=[task], rationale="ok")
    result = TestResult(passed=True, summary="ok", failures=[])
    planner = FakePlanner(plan)
    tester = FakeTester(result)
    client = FakeClient()

    orchestrator = Orchestrator("goal", client=client, planner=planner, tester=tester)
    await orchestrator._cycle()

    assert orchestrator.current_tasks == [task]
    assert task.status == "completed"
    assert orchestrator.last_test_result is result
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_honors_max_cycles():
    plan = make_plan()
    result = TestResult(passed=False, summary="failed", failures=["broken"])
    planner = FakePlanner(plan)
    tester = FakeTester(result)
    client = FakeClient()

    orchestrator = Orchestrator(
        "goal",
        client=client,
        planner=planner,
        tester=tester,
        max_cycles=2,
    )
    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert len(planner.calls) == 2


@pytest.mark.asyncio
async def test_orchestrator_positive_max_cycles_stops_when_tests_pass():
    plan = make_plan()
    result = TestResult(passed=True, summary="ok", failures=[])
    planner = FakePlanner(plan)
    tester = FakeTester(result)
    client = FakeClient()

    orchestrator = Orchestrator(
        "goal",
        client=client,
        planner=planner,
        tester=tester,
        max_cycles=3,
    )
    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert len(planner.calls) == 1
    assert len(tester.calls) == 1


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (Settings(max_cycles=-1), "^max_cycles must be >= 0$"),
        (Settings(max_parallel_tasks=0), "^max_parallel_tasks must be >= 1$"),
    ],
)
@pytest.mark.asyncio
async def test_orchestrator_rejects_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        Orchestrator(
            "goal",
            client=FakeClient(),
            planner=FakePlanner(make_plan()),
            tester=FakeTester(
                TestResult(passed=True, summary="ok", failures=[])
            ),
            settings=settings,
        )


@pytest.mark.asyncio
async def test_orchestrator_honors_max_parallel_tasks():
    plan = Plan(
        tasks=[TaskModel(id=str(index), description=f"task {index}") for index in range(3)],
        rationale="ok",
    )
    result = TestResult(passed=True, summary="ok", failures=[])
    planner = FakePlanner(plan)
    tester = FakeTester(result)
    client = FakeClient()

    orchestrator = Orchestrator(
        "goal",
        client=client,
        planner=planner,
        tester=tester,
        max_parallel_tasks=2,
    )
    await orchestrator.run()

    assert client.max_active == 2


@pytest.mark.asyncio
async def test_orchestrator_preserves_goal_and_passes_failures():
    plan = make_plan()
    result = TestResult(passed=False, summary="failed", failures=["broken test"])
    planner = FakePlanner(plan)
    tester = FakeTester(result)
    client = FakeClient()

    orchestrator = Orchestrator(
        "original goal",
        client=client,
        planner=planner,
        tester=tester,
        max_cycles=2,
    )
    await orchestrator.run()

    assert orchestrator.goal == "original goal"
    assert orchestrator.failures == ["broken test"]
    assert planner.calls[0]["goal"] == "original goal"
    assert planner.calls[0]["failures"] == []
    assert planner.calls[1]["goal"] == "original goal"
    assert planner.calls[1]["failures"] == ["broken test"]


@pytest.mark.asyncio
async def test_orchestrator_stops_on_empty_plan():
    plan = Plan(tasks=[], rationale="done")
    planner = FakePlanner(plan)
    tester = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    client = FakeClient()

    orchestrator = Orchestrator("goal", client=client, planner=planner, tester=tester)
    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert tester.calls == []


class FailingThenEmptyPlanner:
    def __init__(self, fail_plan: Plan, empty_plan: Plan) -> None:
        self.fail_plan = fail_plan
        self.empty_plan = empty_plan
        self.calls: list[dict[str, object]] = []

    async def plan(self, goal: str, failures: list[str] | None = None) -> Plan:
        self.calls.append({"goal": goal, "failures": list(failures or [])})
        if len(self.calls) == 1:
            return self.fail_plan
        return self.empty_plan


@pytest.mark.asyncio
async def test_orchestrator_halts_on_empty_plan_after_failure():
    fail_plan = make_plan()
    empty_plan = Plan(tasks=[], rationale="done")
    planner = FailingThenEmptyPlanner(fail_plan, empty_plan)
    tester = FakeTester(TestResult(passed=False, summary="failed", failures=["broken"]))
    client = FakeClient()

    orchestrator = Orchestrator(
        "original goal",
        client=client,
        planner=planner,
        tester=tester,
        max_cycles=2,
    )
    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator.goal == "original goal"
    assert len(tester.calls) == 1
    assert planner.calls[0]["goal"] == "original goal"
    assert planner.calls[0]["failures"] == []
    assert planner.calls[1]["goal"] == "original goal"
    assert planner.calls[1]["failures"] == ["broken"]


@pytest.mark.asyncio
async def test_planner_prompt_includes_failures():
    class PromptClient:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(planner_model="model")
            self.prompt = ""

        async def complete(self, prompt: str, model: str | None = None) -> str:
            self.prompt = prompt
            return '{"tasks": [], "rationale": "ok"}'

    client = PromptClient()
    planner = PlannerAgent(client=client)

    await planner.plan("fix it", ["first failure", "second failure"])

    assert "Failures to address:" in client.prompt
    assert "- first failure" in client.prompt
    assert "- second failure" in client.prompt


@pytest.mark.asyncio
async def test_planner_plan_remains_backward_compatible():
    class PromptClient:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(planner_model="model")

        async def complete(self, prompt: str, model: str | None = None) -> str:
            return '{"tasks": [], "rationale": "ok"}'

    planner = PlannerAgent(client=PromptClient())

    plan = await planner.plan("goal")

    assert plan.tasks == []

