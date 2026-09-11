import asyncio
import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_get_tasks_returns_plan_tasks():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._get_tasks() == []

    plan = Plan(
        tasks=[TaskModel(id="1", description="task1"), TaskModel(id="2", description="task2")],
        rationale="test",
    )
    orchestrator.plan = plan
    assert orchestrator._get_tasks() == plan.tasks


@pytest.mark.asyncio
async def test_get_tasks_returns_empty_when_no_plan():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._get_tasks() == []


@pytest.mark.asyncio
async def test_is_done_with_failed_tasks():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="failed"),
        ],
        rationale="test",
    )
    orchestrator.plan = plan
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_is_done_with_all_completed():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ],
        rationale="test",
    )
    orchestrator.plan = plan
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_is_done_with_pending_tasks():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(
        tasks=[TaskModel(id="1", description="task1", status="pending")],
        rationale="test",
    )
    orchestrator.plan = plan
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_is_done_with_no_plan():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_max_cycles_enforcement(monkeypatch):
    settings = Settings(max_cycles=2)
    orchestrator = Orchestrator(goal="test", settings=settings)

    call_count = {"cycles": 0}

    plan = Plan(tasks=[TaskModel(id="1", description="task1", status="pending")], rationale="test")

    async def mock_cycle():
        call_count["cycles"] += 1
        orchestrator.plan = plan

    monkeypatch.setattr(orchestrator, "_cycle", mock_cycle)

    await orchestrator.run()

    assert call_count["cycles"] == 2


@pytest.mark.asyncio
async def test_unlimited_max_cycles_with_zero(monkeypatch):
    settings = Settings(max_cycles=0)
    orchestrator = Orchestrator(goal="test", settings=settings)

    call_count = {"cycles": 0}

    async def mock_cycle():
        call_count["cycles"] += 1
        if call_count["cycles"] >= 3:
            orchestrator.plan = Plan(
                tasks=[TaskModel(id="1", description="task1", status="completed")],
                rationale="test",
            )
        else:
            orchestrator.plan = Plan(tasks=[TaskModel(id="1", description="task1", status="pending")], rationale="test")

    monkeypatch.setattr(orchestrator, "_cycle", mock_cycle)

    await orchestrator.run()

    assert call_count["cycles"] == 3


@pytest.mark.asyncio
async def test_max_parallel_tasks_configured():
    settings = Settings(max_parallel_tasks=2)
    orchestrator = Orchestrator(goal="test", settings=settings)
    assert orchestrator.max_parallel_tasks == 2
    assert isinstance(orchestrator._semaphore, asyncio.Semaphore)


@pytest.mark.asyncio
async def test_max_parallel_tasks_zero_defaults_to_one():
    settings = Settings(max_parallel_tasks=0)
    orchestrator = Orchestrator(goal="test", settings=settings)
    assert orchestrator.max_parallel_tasks == 1
    assert isinstance(orchestrator._semaphore, asyncio.Semaphore)


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    settings = Settings(max_parallel_tasks=1)
    orchestrator = Orchestrator(goal="test", settings=settings)

    active = 0
    max_active = 0

    async def tracked_task():
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "done"

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1"),
            TaskModel(id="2", description="task2"),
            TaskModel(id="3", description="task3"),
        ],
        rationale="test",
    )
    orchestrator.plan = plan

    async def tracked_run_with_semaphore(task):
        async with orchestrator._semaphore:
            return await tracked_task()

    orchestrator._run_with_semaphore = tracked_run_with_semaphore

    tasks = [orchestrator._run_with_semaphore(task) for task in plan.tasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert max_active <= 1
    assert all(r == "done" for r in results)


@pytest.mark.asyncio
async def test_custom_settings_applied_to_default_client():
    custom_settings = Settings(max_parallel_tasks=3, provider=Provider.OPENAI)
    orchestrator = Orchestrator(goal="test", settings=custom_settings)
    assert orchestrator.client.settings is custom_settings
    assert orchestrator.client.settings.max_parallel_tasks == 3
    assert orchestrator.client.settings.provider == Provider.OPENAI


@pytest.mark.asyncio
async def test_default_client_uses_global_settings_when_no_custom():
    orchestrator = Orchestrator(goal="test")
    from furrow.config import settings as global_settings
    assert orchestrator.client.settings is global_settings


@pytest.mark.asyncio
async def test_is_done_returns_false_after_failed_test():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ],
        rationale="test",
    )
    orchestrator.plan = plan
    orchestrator.last_test_result = TestResult(
        passed=False, summary="1 failure", failures=["assert x == y"]
    )
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_is_done_returns_true_after_passing_test():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ],
        rationale="test",
    )
    orchestrator.plan = plan
    orchestrator.last_test_result = TestResult(
        passed=True, summary="all passed", failures=[]
    )
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_is_done_returns_true_with_no_test_result_and_completed_tasks():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(
        tasks=[TaskModel(id="1", description="task1", status="completed")],
        rationale="test",
    )
    orchestrator.plan = plan
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_failed_test_allows_retry_after_success():
    orchestrator = Orchestrator(goal="test")
    completed_plan = Plan(
        tasks=[TaskModel(id="1", description="task1", status="completed")],
        rationale="test",
    )
    orchestrator.plan = completed_plan
    orchestrator.last_test_result = TestResult(
        passed=False, summary="fail", failures=["error"]
    )
    assert orchestrator._is_done() is False

    orchestrator.last_test_result = TestResult(
        passed=True, summary="pass", failures=[]
    )
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_failed_test_cleared_by_no_task_plan():
    orchestrator = Orchestrator(goal="test")
    orchestrator.last_test_result = TestResult(
        passed=False, summary="fail", failures=["error"]
    )
    assert orchestrator._is_done() is False

    no_tasks_plan = Plan(tasks=[], rationale="done")
    orchestrator.plan = no_tasks_plan
    orchestrator.last_test_result = None
    assert orchestrator._is_done() is True
