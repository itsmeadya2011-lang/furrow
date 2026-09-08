"""Tests for the Furrow orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.settings = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.workspace = Settings().workspace
    client.settings.worker_model = "test-model"
    client.settings.planner_model = "test-planner"
    client.settings.tester_model = "test-tester"
    client.complete = AsyncMock(return_value="{}")
    client.complete_with_tools = AsyncMock(return_value="mock tool completion")
    client.close = AsyncMock()
    return client


@pytest.fixture
def tmp_settings(tmp_path):
    return Settings(workspace=tmp_path)


class TestOrchestratorTaskTracking:
    """Tests for the orchestrator's task tracking (fixes the _get_tasks bug)."""

    def test_get_tasks_returns_registered_tasks(self):
        orch = Orchestrator(goal="test goal")
        tasks = [TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")]
        for t in tasks:
            orch.tasks[t.id] = t

        result = orch._get_tasks()
        assert len(result) == 2
        ids = {t.id for t in result}
        assert ids == {"1", "2"}

    def test_get_tasks_empty_when_no_tasks(self):
        orch = Orchestrator(goal="test goal")
        assert orch._get_tasks() == []

    def test_is_done_true_when_no_tasks(self):
        orch = Orchestrator(goal="test goal")
        assert orch._is_done() is True

    def test_is_done_true_when_all_completed(self):
        orch = Orchestrator(goal="test goal")
        task = TaskModel(id="1", description="task 1", status="completed")
        orch.tasks["1"] = task
        assert orch._is_done() is True

    def test_is_done_false_when_task_failed(self):
        orch = Orchestrator(goal="test goal")
        task = TaskModel(id="1", description="task 1", status="failed")
        orch.tasks["1"] = task
        assert orch._is_done() is False

    def test_is_done_false_when_task_pending(self):
        orch = Orchestrator(goal="test goal")
        task = TaskModel(id="1", description="task 1", status="pending")
        orch.tasks["1"] = task
        assert orch._is_done() is False

    def test_is_done_false_when_mixed_statuses(self):
        orch = Orchestrator(goal="test goal")
        orch.tasks["1"] = TaskModel(id="1", description="t1", status="completed")
        orch.tasks["2"] = TaskModel(id="2", description="t2", status="pending")
        assert orch._is_done() is False


class TestOrchestratorMaxCycles:
    """Tests for max_cycles enforcement."""

    def test_max_cycles_disabled_by_default(self):
        orch = Orchestrator(goal="test goal")
        orch.cycles = 5
        assert orch._max_cycles_reached() is False

    def test_max_cycles_reached(self, tmp_path):
        settings = Settings(workspace=tmp_path, max_cycles=3)
        orch = Orchestrator(goal="test goal", settings=settings)
        orch.cycles = 3
        assert orch._max_cycles_reached() is True

    def test_max_cycles_not_reached(self, tmp_path):
        settings = Settings(workspace=tmp_path, max_cycles=5)
        orch = Orchestrator(goal="test goal", settings=settings)
        orch.cycles = 2
        assert orch._max_cycles_reached() is False

    def test_max_cycles_zero_is_infinite(self, tmp_path):
        settings = Settings(workspace=tmp_path, max_cycles=0)
        orch = Orchestrator(goal="test goal", settings=settings)
        orch.cycles = 100
        assert orch._max_cycles_reached() is False


class TestOrchestratorRegisterTasks:
    """Tests for task registration from plans."""

    def test_register_tasks_from_plan(self):
        orch = Orchestrator(goal="test goal")
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="task A"),
                TaskModel(id="2", description="task B", dependencies=["1"]),
            ],
            rationale="test plan",
        )
        orch._register_tasks(plan)
        assert "1" in orch.tasks
        assert "2" in orch.tasks
        assert orch.tasks["2"].dependencies == ["1"]

    def test_register_tasks_overwrites(self):
        orch = Orchestrator(goal="test goal")
        plan = Plan(tasks=[TaskModel(id="1", description="task A")], rationale="test")
        orch._register_tasks(plan)
        plan2 = Plan(tasks=[TaskModel(id="1", description="task B")], rationale="test2")
        orch._register_tasks(plan2)
        assert orch.tasks["1"].description == "task B"


class TestOrchestratorStatePersistence:
    """Tests for state save/load."""

    def test_save_and_load_state(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        orch = Orchestrator(goal="my goal", settings=settings)
        orch.cycles = 2
        orch.tasks["1"] = TaskModel(id="1", description="task 1", status="completed")
        orch._save_state()

        orch2 = Orchestrator(goal="my goal", settings=settings)
        orch2._load_state()
        assert orch2.cycles == 2
        assert "1" in orch2.tasks
        assert orch2.tasks["1"].description == "task 1"

    def test_load_state_when_no_file(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        orch = Orchestrator(goal="my goal", settings=settings)
        orch._load_state()
        assert orch.cycles == 0

    def test_state_file_path(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        assert orch._state_file == tmp_path / ".furrow-state.json"

    def test_state_file_created_after_save(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        orch = Orchestrator(goal="test", settings=settings)
        orch.tasks["1"] = TaskModel(id="1", description="task", status="completed")
        orch._save_state()
        assert (tmp_path / ".furrow-state.json").exists()


class TestOrchestratorRun:
    """Tests for the orchestrator's run loop."""

    @pytest.mark.asyncio
    async def test_run_stops_when_done(self, mock_client, tmp_path):
        """Orchestrator should stop when all tasks are completed."""
        settings = Settings(workspace=tmp_path, max_cycles=1)
        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            planner = MockPlanner.return_value
            planner.plan = AsyncMock(return_value=Plan(
                tasks=[TaskModel(id="1", description="task 1")],
                rationale="test",
            ))
            tester = MockTester.return_value
            tester.run = AsyncMock(return_value=TestResult(
                passed=True, summary="All passed", failures=[]
            ))
            orch = Orchestrator(goal="test goal", client=mock_client, settings=settings)
            await orch.run()
            assert orch.cycles == 1
            assert "1" in orch.tasks
            assert orch.tasks["1"].status == "completed"

    @pytest.mark.asyncio
    async def test_run_respects_max_cycles(self, mock_client, tmp_path):
        """Orchestrator should stop when max_cycles is reached, even if not done."""
        settings = Settings(workspace=tmp_path, max_cycles=2)
        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester, \
             patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            planner = MockPlanner.return_value
            planner.plan = AsyncMock(return_value=Plan(
                tasks=[TaskModel(id="1", description="task 1")],
                rationale="test",
            ))
            tester = MockTester.return_value
            tester.run = AsyncMock(return_value=TestResult(
                passed=True, summary="ok", failures=[]
            ))
            # Make worker fail so _is_done returns False
            mock_worker_instance = MockWorker.return_value
            mock_worker_instance.run = AsyncMock(side_effect=Exception("worker error"))

            orch = Orchestrator(goal="test goal", client=mock_client, settings=settings)
            await orch.run()
            assert orch.cycles == 2

    @pytest.mark.asyncio
    async def test_run_continues_on_test_failure(self, mock_client, tmp_path):
        """Orchestrator should continue when tests fail (for next cycle fix)."""
        settings = Settings(workspace=tmp_path, max_cycles=1)
        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            planner = MockPlanner.return_value
            planner.plan = AsyncMock(return_value=Plan(
                tasks=[TaskModel(id="1", description="task 1")],
                rationale="test",
            ))
            tester = MockTester.return_value
            tester.run = AsyncMock(return_value=TestResult(
                passed=False, summary="failures", failures=["test_x failed"]
            ))

            with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"):
                orch = Orchestrator(goal="test goal", client=mock_client, settings=settings)
                await orch.run()
                assert orch.cycles == 1
                # Goal should be updated to reflect test failures
                assert "Fix failing tests" in orch.goal

    @pytest.mark.asyncio
    async def test_run_handles_keyboard_interrupt(self, mock_client, tmp_path):
        """Orchestrator should handle KeyboardInterrupt gracefully."""
        settings = Settings(workspace=tmp_path, max_cycles=10)
        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester, \
             patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            planner = MockPlanner.return_value
            planner.plan = AsyncMock(side_effect=KeyboardInterrupt)
            tester = MockTester.return_value
            tester.run = AsyncMock(return_value=TestResult(
                passed=True, summary="ok", failures=[]
            ))
            mock_worker = MockWorker.return_value
            mock_worker.run = AsyncMock(return_value="done")

            orch = Orchestrator(goal="test goal", client=mock_client, settings=settings)
            await orch.run()  # Should not raise
            assert orch.cycles == 1
