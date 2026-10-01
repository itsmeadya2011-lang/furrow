from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, _should_retry


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# _should_retry helper tests
# ---------------------------------------------------------------------------

def test_should_retry_transient():
    """Transient exceptions should be retried."""
    assert _should_retry(ConnectionError("conn refused"))
    assert _should_retry(TimeoutError("timed out"))
    assert _should_retry(RuntimeError("api error"))


def test_should_retry_non_retryable():
    """Non-retryable exceptions should not be retried."""
    assert not _should_retry(ValueError("bad value"))
    assert not _should_retry(KeyError("missing key"))
    assert not _should_retry(AssertionError("assertion failed"))


def test_should_retry_not_signal():
    """KeyboardInterrupt/SystemExit should not be retried."""
    assert not _should_retry(KeyboardInterrupt())
    assert not _should_retry(SystemExit())


# ---------------------------------------------------------------------------
# Orchestrator tests (use a fake LLMClient so no network is needed)
# ---------------------------------------------------------------------------


class _FakeClient(LLMClient):
    """A lightweight fake that returns canned responses for each model tier."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self._anthropic = None
        self._openai = None
        self._httpx = None
        self.call_count = 0

    async def complete(self, prompt, system="", model=None, messages=None):
        self.call_count += 1
        model = model or self.settings.model
        if "haiku" in model or "planner" in model:
            return '{"tasks": [], "rationale": "nothing to do"}'
        return "done"


@pytest.mark.asyncio
async def test_orchestrator_no_tasks_completes():
    """When the planner returns no tasks, the cycle is considered done."""
    orch = Orchestrator(goal="test", client=_FakeClient())
    await orch._cycle()
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_respected():
    """The loop should stop after max_cycles when set."""
    settings = Settings(max_cycles=2)
    orch = Orchestrator(
        goal="test", client=_FakeClient(settings=settings), settings=settings
    )
    orch._cycle = AsyncMock()
    await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_is_done_after_completion():
    """_is_done should return True only when all tasks completed AND tests passed."""
    orch = Orchestrator(goal="test", client=_FakeClient(), settings=Settings())
    orch._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is True

    # If tests haven't passed yet, not done
    orch._tests_passed = False
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_after_failure():
    """_is_done should return False when any task has failed."""
    orch = Orchestrator(goal="test", client=_FakeClient(), settings=Settings())
    orch._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_dependency_ordering():
    """Tasks with dependencies should run after their dependencies complete."""
    settings = Settings()
    orch = Orchestrator(
        goal="test", client=_FakeClient(settings=settings), settings=settings
    )
    orch._semaphore = asyncio.Semaphore(10)

    tasks = [
        TaskModel(id="a", description="task a", dependencies=["b"]),
        TaskModel(id="b", description="task b", dependencies=[]),
        TaskModel(id="c", description="task c", dependencies=[]),
    ]

    call_order: list[str] = []

    class _RecordingWorker:
        def __init__(self, task, client=None):
            self.task = task

        async def run(self):
            call_order.append(self.task.id)
            return f"result for {self.task.id}"

    with patch("furrow.core.orchestrator.WorkerAgent", _RecordingWorker):
        results = await orch._execute_tasks(tasks)

    # Task 'a' depends on 'b', so 'b' must start before 'a'
    assert call_order.index("b") < call_order.index("a")
    assert len(results) == 3
    for r in results:
        assert not isinstance(r, Exception)


@pytest.mark.asyncio
async def test_orchestrator_circular_dependency():
    """Circular dependencies should be gracefully handled."""
    settings = Settings()
    orch = Orchestrator(
        goal="test", client=_FakeClient(settings=settings), settings=settings
    )
    orch._semaphore = asyncio.Semaphore(10)

    tasks = [
        TaskModel(id="a", description="task a", dependencies=["b"]),
        TaskModel(id="b", description="task b", dependencies=["a"]),
    ]

    class _NoOpWorker:
        def __init__(self, task, client=None):
            self.task = task

        async def run(self):
            return f"done {self.task.id}"

    with patch("furrow.core.orchestrator.WorkerAgent", _NoOpWorker):
        results = await orch._execute_tasks(tasks)

    # Both tasks should have RuntimeError results due to circular dependency
    assert isinstance(results[0], RuntimeError)
    assert isinstance(results[1], RuntimeError)


def test_settings_defaults():
    """Verify default settings values."""
    s = Settings()
    assert s.max_parallel_tasks == 5
    assert s.retry_attempts == 3
    assert s.temperature == 0.7
    assert s.max_tokens == 8192
