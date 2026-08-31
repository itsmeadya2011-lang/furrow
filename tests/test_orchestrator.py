from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def make_client(max_cycles: int = 0):
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = max_cycles
    client.settings.planner_model = "p"
    client.settings.worker_model = "w"
    client.settings.tester_model = "t"
    client.complete = AsyncMock(return_value="{}")
    client.read_file = AsyncMock(return_value="")
    client.write_file = AsyncMock(return_value=None)
    return client


def test_is_done_when_no_tasks():
    client = make_client()
    orch = Orchestrator(goal="build it", client=client)
    assert orch._is_done() is True


def test_is_done_with_pending_tasks():
    client = make_client()
    orch = Orchestrator(goal="build it", client=client)
    orch._tasks = [TaskModel(id="1", description="a", status="pending")]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is False


def test_is_done_with_failed_tasks():
    client = make_client()
    orch = Orchestrator(goal="build it", client=client)
    orch._tasks = [TaskModel(id="1", description="a", status="failed", result="boom")]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is False


def test_is_done_when_all_completed_and_tests_passed():
    client = make_client()
    orch = Orchestrator(goal="build it", client=client)
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch._last_test_result = TestResult(passed=True, summary="all good")
    assert orch._is_done() is True


def test_is_done_false_when_completed_but_tests_failed():
    client = make_client()
    orch = Orchestrator(goal="build it", client=client)
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    orch._last_test_result = TestResult(passed=False, summary="nope", failures=["x failed"])
    assert orch._is_done() is False


def test_get_tasks_returns_stored_tasks():
    client = make_client()
    orch = Orchestrator(goal="build it", client=client)
    stored = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    orch._tasks = stored
    assert orch._get_tasks() is stored
    assert len(orch._get_tasks()) == 2


async def test_max_cycles_enforced():
    client = make_client(max_cycles=2)
    orch = Orchestrator(goal="build it", client=client)

    failing = TestResult(passed=False, summary="bad", failures=["x failed"])
    empty_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=empty_plan)), \
         patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        tester_instance = TesterMock.return_value
        tester_instance.run = AsyncMock(return_value=failing)

        async def fake_execute_tasks():
            for t in orch._tasks:
                t.status = "completed"
        with patch.object(orch, "_execute_tasks", new=fake_execute_tasks):
            await orch.run()

    assert orch.cycles == 2


async def test_on_progress_callback_called():
    client = make_client(max_cycles=1)
    messages: list[str] = []

    def cb(msg: str) -> None:
        messages.append(msg)

    orch = Orchestrator(goal="build it", client=client, on_progress=cb)

    passing = TestResult(passed=True, summary="ok", failures=[])
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)), \
         patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        tester_instance = TesterMock.return_value
        tester_instance.run = AsyncMock(return_value=passing)

        async def fake_execute_tasks():
            for t in orch._tasks:
                t.status = "completed"
        with patch.object(orch, "_execute_tasks", new=fake_execute_tasks):
            await orch.run()

    assert len(messages) > 0
    assert any("Cycle 1" in m for m in messages)
    assert any("Goal complete" in m or "Tests passed" in m for m in messages)