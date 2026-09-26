"""Comprehensive tests for the Furrow orchestrator, state, and config improvements."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.state import FurrowState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state_file(tmp_path: Path) -> Path:
    """Provide a temporary state file path."""
    return tmp_path / ".furrow" / "state.json"


@pytest.fixture
def test_settings(tmp_state_file: Path) -> Settings:
    """Provide Settings with a temp state file and reduced parallelism."""
    return Settings(
        state_file=tmp_state_file,
        max_parallel_tasks=2,
        max_cycles=0,
    )


def make_completed_task(task_id: str, desc: str = "task") -> TaskModel:
    return TaskModel(id=task_id, description=desc, status="completed")


def make_pending_task(task_id: str, desc: str = "task") -> TaskModel:
    return TaskModel(id=task_id, description=desc, status="pending")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_config_ollama_settings():
    s = Settings()
    assert s.ollama_model == "llama3"
    assert s.ollama_base_url == "http://localhost:11434"


def test_config_state_file_default():
    s = Settings()
    assert s.state_file.name == "state.json"


def test_config_max_parallel_tasks_default():
    s = Settings()
    assert s.max_parallel_tasks == 5


def test_config_max_cycles_default():
    s = Settings()
    assert s.max_cycles == 0


# ---------------------------------------------------------------------------
# State (FurrowState) tests
# ---------------------------------------------------------------------------

def test_state_save_load(tmp_state_file: Path):
    state = FurrowState(
        goal="test goal",
        cycles=3,
        all_tasks=[TaskModel(id="1", description="do thing")],
        last_test_result=TestResult(passed=True, summary="ok", failures=[]),
        updated_at=1234567890.0,
    )
    state.save(tmp_state_file)

    loaded = FurrowState.load(tmp_state_file)
    assert loaded is not None
    assert loaded.goal == "test goal"
    assert loaded.cycles == 3
    assert len(loaded.all_tasks) == 1
    assert loaded.all_tasks[0].description == "do thing"
    assert loaded.last_test_result is not None
    assert loaded.last_test_result.passed is True
    assert loaded.updated_at == 1234567890.0


def test_state_load_missing_file(tmp_path: Path):
    result = FurrowState.load(tmp_path / "nonexistent.json")
    assert result is None


def test_state_load_corrupt(tmp_state_file: Path):
    tmp_state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_state_file.write_text("{invalid json")

    result = FurrowState.load(tmp_state_file)
    assert result is None


def test_state_save_creates_parent_dirs(tmp_path: Path):
    state_file = tmp_path / "subdir" / "nested" / "state.json"
    state = FurrowState(goal="test", cycles=0, all_tasks=[], updated_at=0.0)
    state.save(state_file)
    assert state_file.exists()


# ---------------------------------------------------------------------------
# Orchestrator _is_done tests
# ---------------------------------------------------------------------------

async def test_is_done_no_plan(test_settings: Settings):
    """No plan at all means nothing to do = done."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    orch.current_plan = None
    assert orch._is_done() is True


async def test_is_done_empty_plan(test_settings: Settings):
    """Plan with no tasks = done."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    orch.current_plan = Plan(tasks=[], rationale="no work")
    assert orch._is_done() is True


async def test_is_done_all_completed_tests_passed(test_settings: Settings):
    """All tasks completed + tests passed = done."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    orch.current_plan = Plan(
        tasks=[make_completed_task("1")],
        rationale="test",
    )
    orch.last_test_passed = True
    assert orch._is_done() is True


async def test_is_done_tests_failed(test_settings: Settings):
    """Tasks completed but tests failed = not done."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    orch.current_plan = Plan(
        tasks=[make_completed_task("1")],
        rationale="test",
    )
    orch.last_test_passed = False
    assert orch._is_done() is False


async def test_is_done_tasks_pending(test_settings: Settings):
    """Tests passed but tasks still pending = not done."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    orch.current_plan = Plan(
        tasks=[make_pending_task("1")],
        rationale="test",
    )
    orch.last_test_passed = True
    assert orch._is_done() is False


async def test_is_done_one_task_failed(test_settings: Settings):
    """One task failed = not done."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    orch.current_plan = Plan(
        tasks=[
            make_completed_task("1"),
            TaskModel(id="2", description="failed task", status="failed"),
        ],
        rationale="test",
    )
    orch.last_test_passed = True
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
# Orchestrator _get_tasks tests
# ---------------------------------------------------------------------------

async def test_get_tasks_returns_all_tasks(test_settings: Settings):
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    orch.all_tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    assert len(orch._get_tasks()) == 2


# ---------------------------------------------------------------------------
# Orchestrator _execute_tasks tests
# ---------------------------------------------------------------------------

async def test_execute_tasks_independent(test_settings: Settings):
    """Independent tasks should all be completed."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"):
        await orch._execute_tasks(tasks)

    assert all(t.status == "completed" for t in tasks)
    assert all(t.result == "done" for t in tasks)


async def test_execute_tasks_with_dependencies(test_settings: Settings):
    """Task with a dependency should complete after its dependency."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    tasks = [
        TaskModel(id="1", description="dep task", dependencies=[]),
        TaskModel(id="2", description="dependent task", dependencies=["1"]),
    ]

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"):
        await orch._execute_tasks(tasks)

    assert all(t.status == "completed" for t in tasks)


async def test_execute_tasks_circular_dependency(test_settings: Settings):
    """Tasks with circular dependencies should be marked as failed."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    tasks = [
        TaskModel(id="1", description="task 1", dependencies=["2"]),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"):
        await orch._execute_tasks(tasks)

    assert all(t.status == "failed" for t in tasks)


async def test_execute_tasks_worker_exception(test_settings: Settings):
    """A worker exception should mark the task as failed, not crash."""
    orch = Orchestrator(goal="test", custom_settings=test_settings)
    tasks = [TaskModel(id="1", description="task 1")]

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock, side_effect=ValueError("boom")):
        await orch._execute_tasks(tasks)

    assert tasks[0].status == "failed"
    assert "boom" in tasks[0].result


# ---------------------------------------------------------------------------
# Orchestrator max_parallel_tasks (semaphore) test
# ---------------------------------------------------------------------------

async def test_semaphore_limits_concurrency(tmp_path: Path):
    """The semaphore should limit concurrent worker executions."""
    state_file = tmp_path / "state.json"
    settings = Settings(
        state_file=state_file,
        max_parallel_tasks=2,
    )
    orch = Orchestrator(goal="test", custom_settings=settings)

    concurrent = 0
    max_seen = 0

    async def tracking_run(*args, **kwargs):
        nonlocal concurrent, max_seen
        concurrent += 1
        max_seen = max(max_seen, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return "done"

    tasks = [
        TaskModel(id=str(i), description=f"task {i}")
        for i in range(6)
    ]

    with patch.object(WorkerAgent, "run", new_callable=AsyncMock, side_effect=tracking_run):
        await orch._execute_tasks(tasks)

    assert max_seen <= 2  # Never more than max_parallel_tasks
    assert all(t.status == "completed" for t in tasks)


# ---------------------------------------------------------------------------
# Orchestrator max_cycles test
# ---------------------------------------------------------------------------

async def test_max_cycles_enforced(tmp_path: Path):
    """The orchestrator should stop after max_cycles even if tests keep failing."""
    state_file = tmp_path / "state.json"
    test_settings = Settings(
        state_file=state_file,
        max_cycles=2,
        max_parallel_tasks=2,
    )
    orch = Orchestrator(goal="test", custom_settings=test_settings)

    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    failing_test = TestResult(passed=False, summary="fail", failures=["fix this"])

    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan), \
         patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"), \
         patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=failing_test):
        await orch.run()

    assert orch.cycles == 2


async def test_max_cycles_zero_unlimited(tmp_path: Path):
    """max_cycles=0 means the loop stops when tests pass (not after a fixed count)."""
    state_file = tmp_path / "state.json"
    test_settings = Settings(
        state_file=state_file,
        max_cycles=0,
        max_parallel_tasks=2,
    )
    orch = Orchestrator(goal="test", custom_settings=test_settings)

    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    passing_test = TestResult(passed=True, summary="all good", failures=[])

    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan), \
         patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"), \
         patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=passing_test):
        await orch.run()

    assert orch.cycles == 1  # Should stop after one cycle because tests passed


# ---------------------------------------------------------------------------
# Orchestrator _merge_tasks tests
# ---------------------------------------------------------------------------

def test_merge_tasks_new(tmp_path: Path):
    """New tasks should be appended to all_tasks."""
    settings = Settings(state_file=tmp_path / "state.json")
    orch = Orchestrator(goal="test", custom_settings=settings)
    tasks = [TaskModel(id="1", description="new task")]
    orch._merge_tasks(tasks)

    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].id == "1"


def test_merge_tasks_existing(tmp_path: Path):
    """Existing tasks should be updated, not duplicated."""
    settings = Settings(state_file=tmp_path / "state.json")
    orch = Orchestrator(goal="test", custom_settings=settings)
    orch.all_tasks.append(TaskModel(id="1", description="original", status="completed"))

    updated = TaskModel(id="1", description="updated task")
    orch._merge_tasks([updated])

    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].description == "updated task"
    assert orch.all_tasks[0].status == "pending"


# ---------------------------------------------------------------------------
# Orchestrator _save_state tests
# ---------------------------------------------------------------------------

async def test_save_state(tmp_path: Path):
    """_save_state should persist the orchestrator state to disk."""
    state_file = tmp_path / "state.json"
    test_settings = Settings(state_file=state_file)
    orch = Orchestrator(goal="test goal", custom_settings=test_settings)
    orch.cycles = 2
    orch.all_tasks = [TaskModel(id="1", description="task", status="completed")]
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])

    orch._save_state()

    assert state_file.exists()
    loaded = FurrowState.load(state_file)
    assert loaded is not None
    assert loaded.goal == "test goal"
    assert loaded.cycles == 2
    assert len(loaded.all_tasks) == 1


async def test_state_persistence_resumes(tmp_path: Path):
    """The orchestrator should load state from disk on init."""
    state_file = tmp_path / "state.json"
    settings = Settings(state_file=state_file)

    # Write a state file manually
    state = FurrowState(
        goal="persisted goal",
        cycles=5,
        all_tasks=[TaskModel(id="1", description="done", status="completed")],
        last_test_result=TestResult(passed=True, summary="ok", failures=[]),
        updated_at=123456.0,
    )
    state.save(state_file)

    # Create a new orchestrator — it should load the persisted state
    orch = Orchestrator(goal="new goal", custom_settings=settings)

    assert orch.goal == "persisted goal"
    assert orch.cycles == 5
    assert len(orch.all_tasks) == 1
    assert orch.last_test_passed is True


# ---------------------------------------------------------------------------
# Orchestrator _cycle with planning error
# ---------------------------------------------------------------------------

async def test_planning_failure_sets_empty_plan(tmp_path: Path):
    """If planning fails, the orchestrator should not crash."""
    state_file = tmp_path / "state.json"
    test_settings = Settings(
        state_file=state_file,
        max_cycles=1,
        max_parallel_tasks=2,
    )
    orch = Orchestrator(goal="test", custom_settings=test_settings)

    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, side_effect=RuntimeError("LLM error")), \
         patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"), \
         patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=TestResult(passed=True, summary="ok", failures=[])):
        await orch.run()

    # Should have completed 1 cycle without crashing
    assert orch.cycles == 1
    assert orch.current_plan is not None
    assert len(orch.current_plan.tasks) == 0


# ---------------------------------------------------------------------------
# Existing basic tests (kept for backward compatibility)
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
