from unittest.mock import AsyncMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def _make_client():
    client = LLMClient(settings=Settings())
    client.complete = AsyncMock()
    return client


def _make_plan(task_ids):
    return Plan(
        tasks=[TaskModel(id=tid, description=f"task {tid}") for tid in task_ids],
        rationale="test plan",
    )


async def test_runs_cycles_until_all_tasks_complete_and_tests_pass():
    client = _make_client()
    plan = _make_plan(["1", "2"])
    with patch("furrow.core.orchestrator.PlannerAgent") as m_planner, \
         patch("furrow.core.orchestrator.WorkerAgent") as m_worker, \
         patch("furrow.core.orchestrator.TesterAgent") as m_tester, \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=0, max_parallel_tasks=5)):
        m_planner.return_value.plan = AsyncMock(return_value=plan)
        m_worker.return_value.run = AsyncMock(return_value="summary")
        m_tester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orch = Orchestrator(goal="do it", client=client)
        await orch.run()
    assert orch.cycles == 1


async def test_max_cycles_is_enforced():
    client = _make_client()
    plan = _make_plan(["1"])
    with patch("furrow.core.orchestrator.PlannerAgent") as m_planner, \
         patch("furrow.core.orchestrator.WorkerAgent") as m_worker, \
         patch("furrow.core.orchestrator.TesterAgent") as m_tester, \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=2, max_parallel_tasks=5)):
        m_planner.return_value.plan = AsyncMock(return_value=plan)
        m_worker.return_value.run = AsyncMock(return_value="summary")
        m_tester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["x"])
        )
        orch = Orchestrator(goal="do it", client=client)
        await orch.run()
    assert orch.cycles == 2


async def test_done_detection_all_tasks_complete_and_tests_pass():
    client = _make_client()
    orch = Orchestrator(goal="g", client=client)
    orch._current_tasks = [
        TaskModel(id="1", status="completed"),
        TaskModel(id="2", status="completed"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is True


async def test_done_is_false_when_tests_fail():
    client = _make_client()
    orch = Orchestrator(goal="g", client=client)
    orch._current_tasks = [TaskModel(id="1", status="completed")]
    orch._tests_passed = False
    assert orch._is_done() is False


async def test_done_is_false_when_a_task_failed():
    client = _make_client()
    orch = Orchestrator(goal="g", client=client)
    orch._current_tasks = [
        TaskModel(id="1", status="completed"),
        TaskModel(id="2", status="failed"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is False


async def test_loop_continues_when_tests_fail_goal_rewritten():
    client = _make_client()
    plan = _make_plan(["1"])
    failures = ["test_x failed"]
    with patch("furrow.core.orchestrator.PlannerAgent") as m_planner, \
         patch("furrow.core.orchestrator.WorkerAgent") as m_worker, \
         patch("furrow.core.orchestrator.TesterAgent") as m_tester, \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=2, max_parallel_tasks=5)):
        m_planner.return_value.plan = AsyncMock(return_value=plan)
        m_worker.return_value.run = AsyncMock(return_value="summary")
        m_tester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=failures)
        )
        orch = Orchestrator(goal="do it", client=client)
        await orch.run()
    assert orch.cycles == 2
    assert orch.goal.startswith("Fix failing tests:")
    assert "test_x failed" in orch.goal


async def test_loop_continues_when_tasks_raise_exceptions():
    client = _make_client()
    plan = _make_plan(["1"])
    with patch("furrow.core.orchestrator.PlannerAgent") as m_planner, \
         patch("furrow.core.orchestrator.WorkerAgent") as m_worker, \
         patch("furrow.core.orchestrator.TesterAgent") as m_tester, \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=2, max_parallel_tasks=5)):
        m_planner.return_value.plan = AsyncMock(return_value=plan)
        m_worker.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
        m_tester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["boom"])
        )
        orch = Orchestrator(goal="do it", client=client)
        await orch.run()
    assert orch.cycles == 2
    assert plan.tasks[0].status == "failed"
    assert plan.tasks[0].result == "boom"


async def test_max_parallel_tasks_limits_concurrency():
    import asyncio

    client = _make_client()
    plan = _make_plan(["1", "2", "3", "4", "5"])
    max_concurrent = 0
    current = 0
    lock = asyncio.Lock()

    async def fake_run():
        nonlocal max_concurrent, current
        async with lock:
            current += 1
            max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return "done"

    with patch("furrow.core.orchestrator.PlannerAgent") as m_planner, \
         patch("furrow.core.orchestrator.WorkerAgent") as m_worker, \
         patch("furrow.core.orchestrator.TesterAgent") as m_tester, \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=0, max_parallel_tasks=2)):
        m_planner.return_value.plan = AsyncMock(return_value=plan)
        m_worker.return_value.run = AsyncMock(side_effect=fake_run)
        m_tester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orch = Orchestrator(goal="do it", client=client)
        await orch.run()
    assert max_concurrent <= 2


async def test_no_tasks_planned_means_done():
    client = _make_client()
    empty_plan = Plan(tasks=[], rationale="nothing to do")
    with patch("furrow.core.orchestrator.PlannerAgent") as m_planner, \
         patch("furrow.core.orchestrator.WorkerAgent") as m_worker, \
         patch("furrow.core.orchestrator.TesterAgent") as m_tester, \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=0, max_parallel_tasks=5)):
        m_planner.return_value.plan = AsyncMock(return_value=empty_plan)
        m_worker.return_value.run = AsyncMock(return_value="summary")
        m_tester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        orch = Orchestrator(goal="already done", client=client)
        await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True
