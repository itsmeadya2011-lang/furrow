"""Tests for the Orchestrator's fixed behavior: task tracking, max_cycles,
dependency resolution, and state persistence."""

import asyncio
import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class MockLLMClient:
    """Mock LLM client that returns canned responses.

    When the response queue for a given model is empty, the last response
    set for that model is reused so that multiple cycles work without
    re-queuing.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._queue: dict[str, list[str]] = {}
        self._last: dict[str, str] = {}
        self.call_count = 0

    def _set(self, model_key: str, value: str) -> None:
        self._queue[model_key] = [value]
        self._last[model_key] = value

    def set_planner_response(self, plan: Plan) -> None:
        self._set(self.settings.planner_model, json.dumps(plan.model_dump()))

    def set_worker_response(self, response: str) -> None:
        self._set(self.settings.worker_model, response)

    def set_tester_response(self, result: TestResult) -> None:
        self._set(self.settings.tester_model, json.dumps(result.model_dump()))

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        model = model or self.settings.model
        queue = self._queue.get(model, [])
        if queue:
            return queue.pop(0)
        if model in self._last:
            return self._last[model]
        return ""


@pytest.fixture
def tmp_state_dir(tmp_path):
    """Provide a temp directory and state file path."""
    state_dir = tmp_path / ".furrow"
    state_dir.mkdir()
    state_file = state_dir / "state.json"
    return tmp_path, state_file


@pytest.fixture
def settings(tmp_state_dir):
    _, state_file = tmp_state_dir
    s = Settings(
        state_file=state_file,
        max_cycles=0,
        max_parallel_tasks=5,
    )
    return s


@pytest.fixture
def mock_client(settings):
    return MockLLMClient(settings=settings)


@pytest.fixture
def no_real_tests(monkeypatch):
    """Patch TesterAgent._run_tests to avoid running real test suites during tests."""
    async def _mock(self):
        return ""
    monkeypatch.setattr(TesterAgent, "_run_tests", _mock)


# --------------------------------------------------------------------------- #
# _get_tasks / _is_done
# --------------------------------------------------------------------------- #

def test_get_tasks_returns_stored_tasks(settings, mock_client):
    """_get_tasks() should return the tasks stored on the orchestrator."""
    orch = Orchestrator(goal="test", client=mock_client)
    orch._tasks = [TaskModel(id="1", description="task 1")]
    assert len(orch._get_tasks()) == 1
    assert orch._get_tasks()[0].description == "task 1"


def test_is_done_no_tasks(settings, mock_client):
    """With no tasks, _is_done() should return True."""
    orch = Orchestrator(goal="test", client=mock_client)
    assert orch._is_done() is True


def test_is_done_all_completed(settings, mock_client):
    """When all tasks are completed, _is_done() should return True."""
    orch = Orchestrator(goal="test", client=mock_client)
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_with_failed(settings, mock_client):
    """When any task is failed, _is_done() should return False."""
    orch = Orchestrator(goal="test", client=mock_client)
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="failed", result="error"),
    ]
    assert orch._is_done() is False


def test_is_done_with_pending(settings, mock_client):
    """When tasks are still pending, _is_done() should return False."""
    orch = Orchestrator(goal="test", client=mock_client)
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert orch._is_done() is False


# --------------------------------------------------------------------------- #
# _execute_tasks — dependency resolution
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_execute_tasks_respects_dependencies(settings, mock_client):
    """Tasks with unmet dependencies should wait, then execute once deps complete."""
    mock_client.set_worker_response("done")
    orch = Orchestrator(goal="test", client=mock_client)

    tasks = [
        TaskModel(id="1", description="independent task", dependencies=[]),
        TaskModel(id="2", description="depends on task 1", dependencies=["1"]),
    ]

    await orch._execute_tasks(tasks)

    # Both should be completed since task 1 succeeded
    assert tasks[0].status == "completed"
    assert tasks[1].status == "completed"


@pytest.mark.asyncio
async def test_execute_tasks_blocks_on_failed_dependency(settings, mock_client):
    """When a task fails, tasks depending on it should be marked as blocked."""
    orch = Orchestrator(goal="test", client=mock_client)

    original_run = WorkerAgent.run

    async def failing_run(self):
        if self.task.id == "1":
            raise RuntimeError("worker failed")
        return "done"

    WorkerAgent.run = failing_run
    try:
        tasks = [
            TaskModel(id="1", description="failing task", dependencies=[]),
            TaskModel(id="2", description="depends on task 1", dependencies=["1"]),
        ]
        await orch._execute_tasks(tasks)

        assert tasks[0].status == "failed"
        assert tasks[1].status == "failed"
        assert "Blocked" in tasks[1].result
    finally:
        WorkerAgent.run = original_run


@pytest.mark.asyncio
async def test_execute_tasks_concurrency_limit(tmp_state_dir):
    """max_parallel_tasks should limit concurrent execution."""
    _, state_file = tmp_state_dir
    s = Settings(state_file=state_file, max_cycles=0, max_parallel_tasks=2)
    orch = Orchestrator(goal="test", client=MockLLMClient(settings=s))

    max_concurrent = 0
    current_concurrent = 0
    original_run = WorkerAgent.run

    async def tracking_run(self):
        nonlocal max_concurrent, current_concurrent
        current_concurrent += 1
        max_concurrent = max(max_concurrent, current_concurrent)
        await asyncio.sleep(0.1)
        current_concurrent -= 1
        return f"result-{self.task.id}"

    WorkerAgent.run = tracking_run
    try:
        tasks = [
            TaskModel(id=str(i), description=f"task {i}", dependencies=[])
            for i in range(5)
        ]
        await orch._execute_tasks(tasks)
        assert max_concurrent <= 2, f"Expected max 2 concurrent, got {max_concurrent}"
    finally:
        WorkerAgent.run = original_run


@pytest.mark.asyncio
async def test_execute_tasks_resolves_in_waves(settings, mock_client):
    """Tasks in a dependency chain should execute in the correct order (waves)."""
    call_order = []
    orch = Orchestrator(goal="test", client=mock_client)
    original_run = WorkerAgent.run

    async def tracking_run(self):
        call_order.append(self.task.id)
        await asyncio.sleep(0.01)  # ensure ordering is observable
        return f"result-{self.task.id}"

    WorkerAgent.run = tracking_run
    try:
        tasks = [
            TaskModel(id="a", description="a", dependencies=[]),
            TaskModel(id="b", description="b", dependencies=["a"]),
            TaskModel(id="c", description="c", dependencies=["b"]),
        ]
        await orch._execute_tasks(tasks)

        # Task a must come before b, b before c
        assert call_order.index("a") < call_order.index("b") < call_order.index("c")
    finally:
        WorkerAgent.run = original_run


# --------------------------------------------------------------------------- #
# max_cycles enforcement
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_max_cycles_enforced(tmp_state_dir, no_real_tests):
    """Orchestrator should stop after max_cycles cycles even if goal is not done."""
    _, state_file = tmp_state_dir
    s = Settings(state_file=state_file, max_cycles=3, max_parallel_tasks=5)
    client = MockLLMClient(settings=s)

    plan = Plan(tasks=[TaskModel(id="1", description="task 1", dependencies=[])], rationale="one task")
    client.set_planner_response(plan)
    client.set_tester_response(TestResult(passed=False, summary="broken", failures=["fail"]))

    # Make WorkerAgent always raise so the task fails -> _is_done() stays False
    original_run = WorkerAgent.run

    async def failing_run(self):
        raise RuntimeError("always fails")

    WorkerAgent.run = failing_run
    try:
        orch = Orchestrator(goal="test", client=client)
        await orch.run()
        assert orch.cycles == 3
    finally:
        WorkerAgent.run = original_run


# --------------------------------------------------------------------------- #
# State persistence
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_state_save_and_load(tmp_state_dir, no_real_tests):
    """State should be saved to file and loadable on next run."""
    _, state_file = tmp_state_dir
    s = Settings(state_file=state_file, max_cycles=0, max_parallel_tasks=5)
    client = MockLLMClient(settings=s)

    client.set_planner_response(Plan(tasks=[], rationale="done"))
    client.set_tester_response(TestResult(passed=True, summary="ok"))

    orch = Orchestrator(goal="state test", client=client)
    await orch.run()

    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["goal"] == "state test"

    # Load into a new orchestrator
    orch2 = Orchestrator(goal="state test", client=MockLLMClient(settings=s))
    orch2._load_state()
    assert orch2.cycles == orch.cycles


@pytest.mark.asyncio
async def test_state_load_populates_tasks(tmp_state_dir):
    """_load_state() should populate _tasks from the state file."""
    _, state_file = tmp_state_dir
    s = Settings(state_file=state_file, max_cycles=0, max_parallel_tasks=5)

    state = {
        "goal": "loaded goal",
        "cycles": 5,
        "tasks": [
            {"id": "1", "description": "loaded task", "files": [], "dependencies": [], "status": "completed"},
        ],
    }
    state_file.write_text(json.dumps(state))

    orch = Orchestrator(goal="unused", client=MockLLMClient(settings=s))
    orch._load_state()
    assert orch.cycles == 5
    assert len(orch._tasks) == 1
    assert orch._tasks[0].description == "loaded task"
    assert orch._tasks[0].status == "completed"


# --------------------------------------------------------------------------- #
# Full cycle integration
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_cycle_with_passing_tests(settings, mock_client, no_real_tests):
    """A full cycle with passing tests and completed tasks should set _is_done to True."""
    mock_client.set_planner_response(Plan(tasks=[
        TaskModel(id="1", description="task 1", dependencies=[])
    ], rationale="one task"))
    mock_client.set_worker_response("completed task 1")
    mock_client.set_tester_response(TestResult(passed=True, summary="all good"))

    orch = Orchestrator(goal="test goal", client=mock_client)
    await orch._cycle()

    assert len(orch._tasks) == 1
    assert orch._tasks[0].status == "completed"
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_cycle_with_failing_tests_updates_goal(settings, mock_client, no_real_tests):
    """When tests fail, the goal should be updated to fix failing tests."""
    mock_client.set_planner_response(Plan(tasks=[
        TaskModel(id="1", description="task 1", dependencies=[])
    ], rationale="one task"))
    mock_client.set_worker_response("completed task 1")
    mock_client.set_tester_response(TestResult(passed=False, summary="broken", failures=["test_thing"]))

    orch = Orchestrator(goal="original goal", client=mock_client)
    await orch._cycle()

    assert "Fix failing tests" in orch.goal
    assert "test_thing" in orch.goal
