"""Tests for the Orchestrator core loop logic."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Create Settings pointing at a temp workspace."""
    return Settings(workspace=tmp_path, max_cycles=0)


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock LLMClient with async complete."""
    client = MagicMock()
    client.settings = Settings()
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "test"}')
    client.list_files = MagicMock(return_value=[])
    client.read_file = AsyncMock(return_value="")
    client.write_file = AsyncMock(return_value=None)
    return client


class TestIsDone:
    """Tests for the _is_done() completion check."""

    def test_no_tasks_is_done(self, mock_client: MagicMock) -> None:
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch._is_done() is True

    def test_all_completed_is_done(self, mock_client: MagicMock) -> None:
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        assert orch._is_done() is True

    def test_with_failed_tasks_not_done(self, mock_client: MagicMock) -> None:
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ]
        assert orch._is_done() is False

    def test_partial_completion_not_done(self, mock_client: MagicMock) -> None:
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ]
        assert orch._is_done() is False


class TestDependencyExecution:
    """Tests for _execute_with_deps() respecting task dependencies."""

    @pytest.mark.asyncio
    async def test_task_with_no_deps_executes(self, mock_client: MagicMock) -> None:
        orch = Orchestrator(goal="test", client=mock_client)
        tasks = [TaskModel(id="1", description="do thing", status="pending")]
        results = await orch._execute_with_deps(tasks)
        assert len(results) == 1
        assert results[0].status == "completed"

    @pytest.mark.asyncio
    async def test_task_with_satisfied_deps_executes(self, mock_client: MagicMock) -> None:
        orch = Orchestrator(goal="test", client=mock_client)
        tasks = [
            TaskModel(id="1", description="first", status="pending"),
            TaskModel(id="2", description="second", dependencies=["1"], status="pending"),
        ]
        results = await orch._execute_with_deps(tasks)
        assert len(results) == 2
        assert all(t.status == "completed" for t in results)

    @pytest.mark.asyncio
    async def test_dependency_failure_does_not_block_independent(
        self, mock_client: MagicMock
    ) -> None:
        """If a dependency task fails, the dependent task should still be
        attempted (best-effort) after the dependency wave completes."""
        orch = Orchestrator(goal="test", client=mock_client)
        tasks = [
            TaskModel(id="1", description="fail task", status="pending"),
            TaskModel(id="2", description="independent", status="pending"),
            TaskModel(id="3", description="depends on 1", dependencies=["1"], status="pending"),
        ]
        # Mock WorkerAgent class so each instance gets its own side_effect
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            call_count = [0]
            async def side_effect_wrapper(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("boom")
                return "ok"
            instance = AsyncMock(run=side_effect_wrapper)
            MockWorker.return_value = instance
            results = await orch._execute_with_deps(tasks)
        assert results[0].status == "failed"
        assert results[1].status == "completed"
        assert results[2].status == "completed"


class TestStatePersistence:
    """Tests for state save/load."""

    def test_save_and_load_state(self, tmp_path: Path, mock_client: MagicMock) -> None:
        orch = Orchestrator(goal="test", client=mock_client, state_file=tmp_path / ".state.json")
        orch.tasks = [TaskModel(id="1", description="task", status="completed")]
        orch.cycles = 3
        orch._save_state()

        state_file = tmp_path / ".state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["goal"] == "test"
        assert data["cycles"] == 3
        assert len(data["tasks"]) == 1

        # Load into a new orchestrator
        orch2 = Orchestrator(goal="other", client=mock_client, state_file=state_file)
        assert orch2.goal == "test"
        assert orch2.cycles == 3
        assert len(orch2.tasks) == 1


class TestMaxCycles:
    """Tests for the max_cycles limit."""

    @pytest.mark.asyncio
    async def test_max_cycles_stops_loop(self, tmp_path: Path, mock_client: MagicMock) -> None:
        """With max_cycles=1, the loop should run exactly one cycle and stop.

        We mock the planner, worker, and tester to avoid spawning subprocesses
        (which would cause infinite recursion in the test environment).
        """
        from furrow.config import Plan

        settings = Settings(workspace=tmp_path, max_cycles=1)
        mock_client.settings = settings
        orch = Orchestrator(goal="test", client=mock_client, settings=settings)

        with patch.object(orch.planner, "plan", new_callable=AsyncMock) as mock_plan, \
             patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester, \
             patch.object(type(orch), "_save_state", lambda self: None), \
             patch.object(type(orch), "_cleanup_state", lambda self: None):
            mock_plan.return_value = Plan(
                tasks=[TaskModel(id="1", description="t", status="completed")],
                rationale="r",
            )
            MockWorker.return_value.run = AsyncMock(return_value="done")
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )

            await orch.run()

        assert orch.cycles == 1
