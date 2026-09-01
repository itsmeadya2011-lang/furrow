import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(
        tasks=[TaskModel(id="1", description="do thing", files=["a.py"], dependencies=[])],
        rationale="ok",
    )
    assert p.tasks[0].description == "do thing"
    assert p.tasks[0].files == ["a.py"]


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.failures == []


def test_task_model_defaults():
    t = TaskModel(id="x", description="d")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_settings_overrides():
    s = Settings(max_cycles=3, max_parallel_tasks=2, test_timeout_seconds=10)
    assert s.max_cycles == 3
    assert s.max_parallel_tasks == 2
    assert s.test_timeout_seconds == 10


@pytest.mark.asyncio
async def test_orchestrator_is_done_logic(monkeypatch):
    """The orchestrator should not be done with empty or failed task lists."""
    from furrow.core.orchestrator import Orchestrator

    # Empty task list -> not done (need a plan first).
    orch = Orchestrator(goal="anything", settings=Settings(max_cycles=1))
    assert orch._is_done() is False

    # All completed -> done.
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True

    # Any failure -> not done.
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False

    # Any pending -> not done.
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_orchestrator_ready_tasks_respects_dependencies():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="x", settings=Settings())
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", dependencies=[]),
            TaskModel(id="2", description="b", dependencies=["1"]),
            TaskModel(id="3", description="c", dependencies=["2"]),
        ],
        rationale="chain",
    )
    # Nothing completed yet -> only task 1 is ready.
    ready = orch._ready_tasks(plan)
    assert [t.id for t in ready] == ["1"]

    # Mark task 1 completed -> task 2 becomes ready.
    plan.tasks[0].status = "completed"
    orch._tasks = [plan.tasks[0]]
    ready = orch._ready_tasks(plan)
    assert [t.id for t in ready] == ["2"]


def test_orchestrator_stop_flag():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="x", settings=Settings())
    assert orch._stopped is False
    orch.stop()
    assert orch._stopped is True


def test_planner_strips_markdown_fences():
    from furrow.agents.planner import PlannerAgent

    captured: dict[str, str] = {}

    class _Stub:
        settings = Settings()

        async def complete(self, prompt, model=None):
            captured["prompt"] = prompt
            return "```json\n{\"tasks\": [], \"rationale\": \"ok\"}\n```"

    async def run() -> None:
        agent = PlannerAgent(client=_Stub())  # type: ignore[arg-type]
        plan = await agent.plan("anything")
        assert plan.tasks == []
        assert plan.rationale == "ok"

    import asyncio

    asyncio.run(run())
    assert "Goal: anything" in captured["prompt"]
