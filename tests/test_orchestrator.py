from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_orchestrator(tasks=None, settings=None, callback=None):
    client = MagicMock()
    client.settings = settings or Settings(max_cycles=0)
    orch = Orchestrator(goal="test goal", client=client, output_callback=callback)
    if tasks is not None:
        orch.tasks = tasks
    return orch


def test_get_tasks_returns_tasks():
    orch = _make_orchestrator()
    assert orch.tasks == []
    assert orch._get_tasks() is orch.tasks
    assert orch._get_tasks() == []


def test_is_done_no_tasks():
    orch = _make_orchestrator()
    assert orch._is_done() is True


def test_is_done_all_completed():
    tasks = [TaskModel(id="1", description="a", status="completed")]
    orch = _make_orchestrator(tasks=tasks)
    assert orch._is_done() is True


def test_is_done_has_pending():
    tasks = [TaskModel(id="1", description="a", status="pending")]
    orch = _make_orchestrator(tasks=tasks)
    assert orch._is_done() is False


def test_is_done_has_failed():
    tasks = [TaskModel(id="1", description="a", status="failed")]
    orch = _make_orchestrator(tasks=tasks)
    assert orch._is_done() is False


def test_build_waves_no_dependencies():
    orch = _make_orchestrator()
    tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ]
    waves = orch._build_waves(tasks)
    assert len(waves) == 1
    assert len(waves[0]) == 2
    assert {t.id for t in waves[0]} == {"1", "2"}


def test_build_waves_with_dependencies():
    orch = _make_orchestrator()
    t1 = TaskModel(id="1", description="a")
    t2 = TaskModel(id="2", description="b", dependencies=["1"])
    t3 = TaskModel(id="3", description="c", dependencies=["2"])
    waves = orch._build_waves([t1, t2, t3])
    assert len(waves) == 3
    assert [t.id for t in waves[0]] == ["1"]
    assert [t.id for t in waves[1]] == ["2"]
    assert [t.id for t in waves[2]] == ["3"]


def test_build_waves_circular_dependency():
    orch = _make_orchestrator()
    t1 = TaskModel(id="1", description="a", dependencies=["2"])
    t2 = TaskModel(id="2", description="b", dependencies=["1"])
    waves = orch._build_waves([t1, t2])
    assert waves == []
    assert t1.status == "failed"
    assert t1.result == "Unresolvable dependencies"
    assert t2.status == "failed"
    assert t2.result == "Unresolvable dependencies"


def test_build_waves_skips_already_completed():
    orch = _make_orchestrator()
    orch.tasks = [TaskModel(id="1", description="done", status="completed")]
    t1 = TaskModel(id="1", description="done")
    t2 = TaskModel(id="2", description="new", dependencies=["1"])
    waves = orch._build_waves([t1, t2])
    assert len(waves) == 1
    ids_in_waves = {t.id for wave in waves for t in wave}
    assert "1" not in ids_in_waves
    assert "2" in ids_in_waves


async def test_max_cycles_enforced():
    callback = AsyncMock()
    settings = Settings(max_cycles=1)
    client = MagicMock()
    client.settings = settings
    orch = Orchestrator(goal="test", client=client, output_callback=callback)
    orch.tasks = [TaskModel(id="1", description="never completes")]
    orch._cycle = AsyncMock()
    await orch.run()
    assert orch.cycles == 1
    orch._cycle.assert_awaited_once()
    assert orch._is_done() is False


async def test_run_emits_messages():
    callback = AsyncMock()
    settings = Settings(max_cycles=10)
    client = MagicMock()
    client.settings = settings
    orch = Orchestrator(goal="build a thing", client=client, output_callback=callback)
    orch.planner.plan = AsyncMock(
        return_value=Plan(tasks=[TaskModel(id="1", description="task one")], rationale="rationale")
    )
    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="task done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all green", failures=[])
        )
        await orch.run()
    assert orch.cycles == 1
    assert callback.await_count > 0
