from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock()
    client.settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
    return client


@pytest.fixture
def orchestrator(client: MagicMock) -> Orchestrator:
    return Orchestrator(goal="Write comprehensive tests", client=client)


async def test_get_tasks_returns_stored_tasks(orchestrator: Orchestrator):
    task = TaskModel(id="1", description="Write tests")
    orchestrator._tasks = [task]
    assert orchestrator._get_tasks() == [task]


async def test_is_done_no_tasks(orchestrator: Orchestrator):
    orchestrator._tasks = []
    orchestrator._last_test_result = None
    assert orchestrator._is_done() is True


async def test_is_done_all_completed(orchestrator: Orchestrator):
    orchestrator._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
        TaskModel(id="2", description="task two", status="completed"),
    ]
    orchestrator._last_test_result = TestResult(passed=True, summary="all good", failures=[])
    assert orchestrator._is_done() is True


async def test_is_done_tests_failed(orchestrator: Orchestrator):
    orchestrator._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
    ]
    orchestrator._last_test_result = TestResult(
        passed=False, summary="tests failed", failures=["assertion error"]
    )
    assert orchestrator._is_done() is False


async def test_is_done_any_failed(orchestrator: Orchestrator):
    orchestrator._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
        TaskModel(id="2", description="task two", status="failed"),
    ]
    assert orchestrator._is_done() is False


async def test_max_cycles_enforcement(client: MagicMock):
    client.settings = Settings(
        provider=Provider.ANTHROPIC, anthropic_api_key="test", max_cycles=2
    )
    orchestrator = Orchestrator(goal="Write tests", client=client)
    orchestrator._cycle = AsyncMock()
    orchestrator._is_done = MagicMock(return_value=False)

    await orchestrator.run()

    assert orchestrator._cycle.call_count == 2


async def test_worker_retry_succeeds(orchestrator: Orchestrator):
    task = TaskModel(id="1", description="task one")
    with (
        patch("furrow.core.orchestrator.WorkerAgent") as mock_worker,
        patch(
            "furrow.core.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
    ):
        mock_worker.return_value.run = AsyncMock(
            side_effect=[RuntimeError("fail"), RuntimeError("fail"), "ok"]
        )
        result = await orchestrator._run_worker_with_retry(task)

    assert result == "ok"
    assert mock_worker.return_value.run.call_count == 3
    assert mock_sleep.call_count == 2


async def test_worker_retry_exhausted(orchestrator: Orchestrator):
    task = TaskModel(id="1", description="task one")
    with (
        patch("furrow.core.orchestrator.WorkerAgent") as mock_worker,
        patch("furrow.core.orchestrator.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_worker.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            await orchestrator._run_worker_with_retry(task)

    assert mock_worker.return_value.run.call_count == 4
