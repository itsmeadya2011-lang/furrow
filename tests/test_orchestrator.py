from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        settings = Settings(anthropic_api_key="test-key", max_parallel_tasks=2, max_cycles=3)
        client = LLMClient(settings=settings)
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="Test goal", client=mock_client)

    @pytest.mark.asyncio
    async def test_orchestrator_with_mocked_agents(self, orchestrator):
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="Task 1"),
                TaskModel(id="2", description="Task 2"),
            ],
            rationale="Test plan",
        )

        test_result = TestResult(passed=True, summary="All passed", failures=[])

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = plan
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker:
                mock_worker.return_value = "Task completed"
                with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
                    mock_tester.return_value = test_result

                    await orchestrator.run()

        assert mock_plan.called
        assert mock_worker.call_count == 2
        assert mock_tester.called

    @pytest.mark.asyncio
    async def test_task_tracking_across_cycles(self, mock_client):
        """Test the bug fix: task tracking persists across cycles."""
        settings = Settings(anthropic_api_key="test-key", max_cycles=2, max_parallel_tasks=2)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="Test goal", client=client)

        plan1 = Plan(
            tasks=[TaskModel(id="1", description="Task 1")],
            rationale="Plan 1",
        )
        plan2 = Plan(
            tasks=[TaskModel(id="2", description="Task 2")],
            rationale="Plan 2",
        )

        test_result_fail = TestResult(passed=False, summary="Failed", failures=["test failed"])
        test_result_pass = TestResult(passed=True, summary="Passed", failures=[])

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.side_effect = [plan1, plan2]
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker:
                mock_worker.side_effect = Exception("Task failed")
                with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
                    mock_tester.side_effect = [test_result_fail, test_result_pass]

                    await orchestrator.run()

        # Verify tasks from the last cycle were tracked
        assert len(orchestrator._current_tasks) == 1  # Second cycle's tasks
        assert orchestrator._current_tasks[0].id == "2"
        # The goal should be updated after first cycle failure
        assert "Fix failing tests" in orchestrator.goal

    @pytest.mark.asyncio
    async def test_max_cycles_enforcement(self, mock_client):
        settings = Settings(anthropic_api_key="test-key", max_cycles=2, max_parallel_tasks=1)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="Test goal", client=client)

        plan = Plan(tasks=[TaskModel(id="1", description="Task")], rationale="Plan")
        test_result = TestResult(passed=False, summary="Failed", failures=["fail"])

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = plan
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker:
                mock_worker.side_effect = Exception("Task failed")
                with patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
                    mock_tester.return_value = test_result

                    await orchestrator.run()

        # Should stop after 2 cycles
        assert orchestrator.cycles == 2
        assert mock_plan.call_count == 2

    @pytest.mark.asyncio
    async def test_is_done_all_completed(self, orchestrator):
        orchestrator._current_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_some_failed(self, orchestrator):
        orchestrator._current_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_some_pending(self, orchestrator):
        orchestrator._current_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="pending"),
        ]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_empty_tasks(self, orchestrator):
        orchestrator._current_tasks = []
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_topological_sort(self, orchestrator):
        tasks = [
            TaskModel(id="1", description="Task 1", dependencies=[]),
            TaskModel(id="2", description="Task 2", dependencies=["1"]),
            TaskModel(id="3", description="Task 3", dependencies=["1"]),
            TaskModel(id="4", description="Task 4", dependencies=["2", "3"]),
        ]

        ordered = orchestrator._topological_sort(tasks)
        ordered_ids = [t.id for t in ordered]

        # 1 must come before 2 and 3
        assert ordered_ids.index("1") < ordered_ids.index("2")
        assert ordered_ids.index("1") < ordered_ids.index("3")
        # 2 and 3 must come before 4
        assert ordered_ids.index("2") < ordered_ids.index("4")
        assert ordered_ids.index("3") < ordered_ids.index("4")

    @pytest.mark.asyncio
    async def test_topological_sort_cycle_handling(self, orchestrator):
        tasks = [
            TaskModel(id="1", description="Task 1", dependencies=["2"]),
            TaskModel(id="2", description="Task 2", dependencies=["1"]),
        ]

        ordered = orchestrator._topological_sort(tasks)
        # Should still return both tasks even with cycle
        assert len(ordered) == 2

    @pytest.mark.asyncio
    async def test_callback_on_log_called(self, mock_client):
        log_events = []

        def on_log(event, data):
            log_events.append((event, data))

        orchestrator = Orchestrator(goal="Test goal", client=mock_client, on_log=on_log)

        plan = Plan(tasks=[TaskModel(id="1", description="Task")], rationale="Plan")
        test_result = TestResult(passed=True, summary="Passed", failures=[])

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = plan
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock), patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_tester:
                mock_tester.return_value = test_result
                await orchestrator.run()

        # Check that events were emitted
        event_names = [e[0] for e in log_events]
        assert "orchestrator_start" in event_names
        assert "cycle_start" in event_names
        assert "plan_created" in event_names
        assert "goal_complete" in event_names

    @pytest.mark.asyncio
    async def test_emit_forwards_to_structlog_and_callback(self, mock_client):
        log_events = []

        def on_log(event, data):
            log_events.append((event, data))

        orchestrator = Orchestrator(goal="Test", client=mock_client, on_log=on_log)

        with patch("furrow.core.orchestrator.log") as mock_log:
            orchestrator._emit("custom_event", level="warning", key="value")

            mock_log.warning.assert_called_once_with("custom_event", key="value")
            assert ("custom_event", {"key": "value"}) in log_events

    @pytest.mark.asyncio
    async def test_send_forwards_to_callback(self, mock_client):
        messages = []

        def on_log(event, data):
            messages.append((event, data))

        orchestrator = Orchestrator(goal="Test", client=mock_client, on_log=on_log)
        orchestrator._send("Hello world")

        assert ("message", {"message": "Hello world"}) in messages

    @pytest.mark.asyncio
    async def test_no_callback_no_error(self, mock_client):
        orchestrator = Orchestrator(goal="Test", client=mock_client)  # No on_log

        # Should not raise
        orchestrator._emit("event", key="value")
        orchestrator._send("message")

    @pytest.mark.asyncio
    async def test_dependency_ordering_in_execution(self, mock_client):
        orchestrator = Orchestrator(goal="Test", client=mock_client)

        tasks = [
            TaskModel(id="1", description="Task 1", dependencies=[]),
            TaskModel(id="2", description="Task 2", dependencies=["1"]),
            TaskModel(id="3", description="Task 3", dependencies=["1"]),
        ]

        ordered = orchestrator._topological_sort(tasks)
        assert [t.id for t in ordered] == ["1", "2", "3"] or [t.id for t in ordered] == ["1", "3", "2"]

    @pytest.mark.asyncio
    async def test_cycle_increments_and_stops_on_completion(self, mock_client):
        settings = Settings(anthropic_api_key="test-key", max_cycles=5)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="Test", client=client)

        plan = Plan(tasks=[], rationale="Done")
        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = plan
            await orchestrator.run()

        assert orchestrator.cycles == 1