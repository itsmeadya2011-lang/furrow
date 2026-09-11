import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.core.orchestrator import Orchestrator
from furrow.config import TaskModel, TestResult, Settings, Plan
from furrow.llm import LLMClient


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(max_cycles=3)
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="test goal", client=mock_client)

    def test_get_tasks_returns_self_tasks(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1")
        task2 = TaskModel(id="2", description="task 2")
        orchestrator._tasks = [task1, task2]

        result = orchestrator._get_tasks()

        assert result == [task1, task2]
        assert result is orchestrator._tasks

    def test_get_tasks_returns_empty_list_when_no_tasks(self, orchestrator):
        orchestrator._tasks = []

        result = orchestrator._get_tasks()

        assert result == []

    def test_is_done_returns_false_when_tasks_empty(self, orchestrator):
        orchestrator._tasks = []
        orchestrator._tests_passed = True

        result = orchestrator._is_done()

        assert result is False

    def test_is_done_returns_false_when_any_task_failed(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="failed")
        orchestrator._tasks = [task1, task2]
        orchestrator._tests_passed = True

        result = orchestrator._is_done()

        assert result is False

    def test_is_done_returns_false_when_not_all_completed(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="pending")
        orchestrator._tasks = [task1, task2]
        orchestrator._tests_passed = True

        result = orchestrator._is_done()

        assert result is False

    def test_is_done_returns_false_when_all_completed_but_tests_not_passed(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="completed")
        orchestrator._tasks = [task1, task2]
        orchestrator._tests_passed = False

        result = orchestrator._is_done()

        assert result is False

    def test_is_done_returns_true_when_all_completed_and_tests_passed(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="completed")
        orchestrator._tasks = [task1, task2]
        orchestrator._tests_passed = True

        result = orchestrator._is_done()

        assert result is True

    def test_is_done_returns_false_when_some_completed_some_pending_tests_passed(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="pending")
        orchestrator._tasks = [task1, task2]
        orchestrator._tests_passed = True

        result = orchestrator._is_done()

        assert result is False


class TestOrchestratorMaxCycles:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(max_cycles=2)
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="test goal", client=mock_client)

    @pytest.mark.asyncio
    async def test_max_cycles_respected(self, orchestrator):
        """Test that the orchestrator stops after max_cycles."""
        call_count = 0

        async def mock_cycle():
            nonlocal call_count
            call_count += 1

        with patch.object(orchestrator, "_cycle", mock_cycle):
            with patch.object(orchestrator, "_is_done", return_value=False):
                await orchestrator.run()

        assert call_count == 2
        assert orchestrator.cycles == 2


class TestOrchestratorProgressCallback:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(max_cycles=0)
        return client

    @pytest.mark.asyncio
    async def test_progress_callback_called_with_events(self, mock_client):
        events = []

        def callback(event, data):
            events.append((event, data))

        orchestrator = Orchestrator(goal="test goal", client=mock_client, progress_callback=callback)

        async def mock_cycle():
            orchestrator._tasks = [TaskModel(id="1", description="task 1", status="completed")]
            orchestrator._tests_passed = True

        with patch.object(orchestrator, "_cycle", mock_cycle):
            with patch.object(orchestrator, "_is_done", return_value=True):
                await orchestrator.run()

        event_names = [e[0] for e in events]
        assert "cycle_start" in event_names
        assert "goal_complete" in event_names
        assert event_names.count("cycle_start") == 1

    @pytest.mark.asyncio
    async def test_progress_callback_not_called_when_none(self, mock_client):
        orchestrator = Orchestrator(goal="test goal", client=mock_client, progress_callback=None)

        async def mock_cycle():
            orchestrator._tasks = [TaskModel(id="1", description="task 1", status="completed")]
            orchestrator._tests_passed = True

        with patch.object(orchestrator, "_cycle", mock_cycle):
            with patch.object(orchestrator, "_is_done", return_value=True):
                await orchestrator.run()

        assert orchestrator.progress_callback is None


class TestOrchestratorCycle:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(max_cycles=0)
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="test goal", client=mock_client)

    @pytest.mark.asyncio
    async def test_cycle_plans_and_emits_plan(self, orchestrator):
        mock_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test rationale"
        )
        
        async def mock_plan_method(goal):
            return mock_plan
        
        orchestrator.planner.plan = mock_plan_method
        
        events = []
        def callback(event, data):
            events.append((event, data))
        orchestrator.progress_callback = callback

        async def mock_worker_run():
            return "result"
        
        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.run = mock_worker_run
            mock_worker_class.return_value = mock_worker
            
            async def mock_tester_run(goal, tasks):
                return TestResult(passed=True, summary="ok", failures=[])
            
            with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
                mock_tester = AsyncMock()
                mock_tester.run = mock_tester_run
                mock_tester_class.return_value = mock_tester
                
                await orchestrator._cycle()

        assert orchestrator._tasks == mock_plan.tasks
        assert ("plan", mock_plan) in events

    @pytest.mark.asyncio
    async def test_cycle_handles_no_tasks(self, orchestrator):
        mock_plan = Plan(tasks=[], rationale="no tasks needed")
        
        async def mock_plan_method(goal):
            return mock_plan
        
        orchestrator.planner.plan = mock_plan_method
        
        events = []
        def callback(event, data):
            events.append((event, data))
        orchestrator.progress_callback = callback

        await orchestrator._cycle()

        assert orchestrator._tasks == []
        assert ("plan", mock_plan) in events

    @pytest.mark.asyncio
    async def test_cycle_marks_failed_tasks(self, orchestrator):
        mock_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test"
        )
        
        async def mock_plan_method(goal):
            return mock_plan
        
        orchestrator.planner.plan = mock_plan_method
        
        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.run = AsyncMock(side_effect=Exception("Worker failed"))
            mock_worker_class.return_value = mock_worker
            
            async def mock_tester_run(goal, tasks):
                return TestResult(passed=False, summary="failed", failures=["error"])
            
            with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
                mock_tester = AsyncMock()
                mock_tester.run = mock_tester_run
                mock_tester_class.return_value = mock_tester
                
                await orchestrator._cycle()

        assert orchestrator._tasks[0].status == "failed"
        assert "Worker failed" in orchestrator._tasks[0].result