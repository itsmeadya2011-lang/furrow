import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.fixture
def mock_settings():
    return Settings(max_cycles=0)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def orchestrator(mock_client, mock_settings):
    return Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)


class TestOrchestratorIsDone:
    def test_no_plan_returns_false(self, orchestrator):
        assert orchestrator._is_done() is False

    def test_empty_plan_no_test_result_returns_false(self, orchestrator):
        orchestrator.current_plan = Plan(tasks=[], rationale="empty")
        assert orchestrator._is_done() is False

    def test_empty_plan_with_passed_test_returns_true(self, orchestrator):
        orchestrator.current_plan = Plan(tasks=[], rationale="empty")
        orchestrator.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orchestrator._is_done() is True

    def test_empty_plan_with_failed_test_returns_false(self, orchestrator):
        orchestrator.current_plan = Plan(tasks=[], rationale="empty")
        orchestrator.last_test_result = TestResult(passed=False, summary="fail", failures=["error"])
        assert orchestrator._is_done() is False

    def test_pending_tasks_returns_false(self, orchestrator):
        orchestrator.current_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1", status="pending")],
            rationale="test"
        )
        orchestrator.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orchestrator._is_done() is False

    def test_failed_tasks_returns_false(self, orchestrator):
        orchestrator.current_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1", status="failed", result="error")],
            rationale="test"
        )
        orchestrator.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orchestrator._is_done() is False

    def test_all_completed_tasks_passed_test_returns_true(self, orchestrator):
        orchestrator.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1", status="completed", result="done"),
                TaskModel(id="2", description="task 2", status="completed", result="done"),
            ],
            rationale="test"
        )
        orchestrator.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orchestrator._is_done() is True

    def test_all_completed_tasks_failed_test_returns_false(self, orchestrator):
        orchestrator.current_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1", status="completed", result="done")],
            rationale="test"
        )
        orchestrator.last_test_result = TestResult(passed=False, summary="fail", failures=["error"])
        assert orchestrator._is_done() is False

    def test_no_test_result_returns_false(self, orchestrator):
        orchestrator.current_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1", status="completed", result="done")],
            rationale="test"
        )
        orchestrator.last_test_result = None
        assert orchestrator._is_done() is False


class TestOrchestratorGetTasks:
    def test_no_plan_returns_empty_list(self, orchestrator):
        assert orchestrator._get_tasks() == []

    def test_returns_plan_tasks(self, orchestrator):
        tasks = [TaskModel(id="1", description="task 1"), TaskModel(id="2", description="task 2")]
        orchestrator.current_plan = Plan(tasks=tasks, rationale="test")
        assert orchestrator._get_tasks() == tasks


class TestOrchestratorMaxCycles:
    @pytest.mark.asyncio
    async def test_max_cycles_stops_at_limit(self, mock_client, mock_settings):
        mock_settings.max_cycles = 2
        orchestrator = Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)

        with patch.object(orchestrator, '_cycle', new_callable=AsyncMock) as mock_cycle:
            with patch.object(orchestrator, '_is_done', return_value=False):
                with patch('furrow.core.orchestrator.console.print') as mock_print:
                    await orchestrator.run()
                    assert mock_cycle.call_count == 2
                    assert any("Max cycles (2) reached" in str(call) for call in mock_print.call_args_list)

    @pytest.mark.asyncio
    async def test_max_cycles_zero_runs_until_done(self, mock_client, mock_settings):
        mock_settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)

        with patch.object(orchestrator, '_cycle', new_callable=AsyncMock) as mock_cycle:
            is_done_calls = [False, False, True]
            with patch.object(orchestrator, '_is_done', side_effect=is_done_calls):
                with patch('furrow.core.orchestrator.console.print'):
                    await orchestrator.run()
                    assert mock_cycle.call_count == 3


class TestOrchestratorTaskStatusPreservation:
    @pytest.mark.asyncio
    async def test_task_status_preserved_across_cycles(self, mock_client, mock_settings):
        orchestrator = Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)

        task1 = TaskModel(id="1", description="task 1", status="completed", result="done")
        task2 = TaskModel(id="2", description="task 2", status="pending")
        orchestrator.current_plan = Plan(tasks=[task1, task2], rationale="initial")

        new_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1"),
                TaskModel(id="2", description="task 2"),
                TaskModel(id="3", description="task 3"),
            ],
            rationale="new"
        )

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock, return_value=new_plan):
            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                mock_worker.return_value.run = AsyncMock(return_value="done")
                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester.return_value.run = AsyncMock(
                        return_value=TestResult(passed=True, summary="ok", failures=[])
                    )
                    with patch('furrow.core.orchestrator.console.print'):
                        await orchestrator._cycle()

        assert orchestrator.current_plan is not None
        tasks = orchestrator.current_plan.tasks
        task1_updated = next(t for t in tasks if t.id == "1")
        task2_updated = next(t for t in tasks if t.id == "2")
        task3_updated = next(t for t in tasks if t.id == "3")

        assert task1_updated.status == "completed"
        assert task1_updated.result == "done"
        assert task2_updated.status == "pending"
        assert task3_updated.status == "pending"

    @pytest.mark.asyncio
    async def test_failed_task_reset_to_pending_for_retry(self, mock_client, mock_settings):
        orchestrator = Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)

        task1 = TaskModel(id="1", description="task 1", status="completed", result="done")
        task2 = TaskModel(id="2", description="task 2", status="failed", result="boom")
        orchestrator.current_plan = Plan(tasks=[task1, task2], rationale="initial")

        new_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1"),
                TaskModel(id="2", description="task 2"),
            ],
            rationale="new"
        )

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock, return_value=new_plan):
            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                mock_worker.return_value.run = AsyncMock(return_value="done")
                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester.return_value.run = AsyncMock(
                        return_value=TestResult(passed=True, summary="ok", failures=[])
                    )
                    with patch('furrow.core.orchestrator.console.print'):
                        await orchestrator._cycle()

        tasks = orchestrator.current_plan.tasks
        task1_updated = next(t for t in tasks if t.id == "1")
        task2_updated = next(t for t in tasks if t.id == "2")

        assert task1_updated.status == "completed"
        assert task1_updated.result == "done"
        assert task2_updated.status == "pending"
        assert task2_updated.result is None

    @pytest.mark.asyncio
    async def test_failed_test_resets_completed_tasks_for_retry(self, mock_client, mock_settings):
        orchestrator = Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)
        task1 = TaskModel(id="1", description="task 1", status="completed", result="done")
        task2 = TaskModel(id="2", description="task 2", status="completed", result="done")
        orchestrator.current_plan = Plan(tasks=[task1, task2], rationale="initial")
        new_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1"),
                TaskModel(id="2", description="task 2"),
            ],
            rationale="new"
        )

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock, return_value=new_plan):
            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester.return_value.run = AsyncMock(
                        return_value=TestResult(
                            passed=False,
                            summary="tests failed",
                            failures=["fixture failure"],
                        )
                    )
                    with patch('furrow.core.orchestrator.console.print'):
                        await orchestrator._cycle()

        assert all(task.status == "pending" for task in orchestrator.current_plan.tasks)
        assert all(task.result is None for task in orchestrator.current_plan.tasks)
        assert "fixture failure" in orchestrator.goal
        assert mock_worker.call_count == 0


class TestOrchestratorEmptyPlan:
    @pytest.mark.asyncio
    async def test_empty_plan_runs_tester_and_terminates(self, mock_client, mock_settings):
        orchestrator = Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)

        empty_plan = Plan(tasks=[], rationale="empty")
        orchestrator.current_plan = Plan(tasks=[], rationale="empty")

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock, return_value=empty_plan):
            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester.return_value.run = AsyncMock(
                        return_value=TestResult(passed=True, summary="ok", failures=[])
                    )
                    with patch('furrow.core.orchestrator.console.print'):
                        await orchestrator._cycle()

        assert mock_worker.called is False
        assert orchestrator.last_test_result is not None
        assert orchestrator.last_test_result.passed is True
        assert orchestrator._is_done() is True


class TestOrchestratorSettings:
    def test_explicit_settings_used(self, mock_client):
        custom_settings = Settings(max_cycles=5)
        orchestrator = Orchestrator(goal="test", client=mock_client, settings=custom_settings)
        assert orchestrator.settings is custom_settings
        assert orchestrator.settings.max_cycles == 5

    def test_explicit_settings_authoritative_for_injected_client(self, mock_client):
        """When both client and settings are provided, explicit settings must be authoritative for client too."""
        custom_settings = Settings(max_cycles=5, model="custom-model")
        orchestrator = Orchestrator(goal="test", client=mock_client, settings=custom_settings)
        assert orchestrator.settings is custom_settings
        assert orchestrator.client.settings is custom_settings
        assert orchestrator.client.settings.max_cycles == 5
        assert orchestrator.client.settings.model == "custom-model"

    def test_explicit_settings_shared_with_default_client(self):
        custom_settings = Settings(max_cycles=5)
        orchestrator = Orchestrator(goal="test", settings=custom_settings)
        assert orchestrator.settings is custom_settings
        assert orchestrator.client.settings is custom_settings
        assert orchestrator.settings.max_cycles == 5

    def test_client_settings_used_when_no_explicit_settings(self):
        custom_settings = Settings(max_cycles=7)
        client = MagicMock(spec=LLMClient)
        client.settings = custom_settings
        orchestrator = Orchestrator(goal="test", client=client)
        assert orchestrator.settings is custom_settings
        assert orchestrator.settings.max_cycles == 7

    def test_default_settings_for_default_client(self):
        orchestrator = Orchestrator(goal="test")
        assert orchestrator.settings is orchestrator.client.settings


class TestOrchestratorUnknownStatus:
    def test_unknown_status_treated_as_pending(self, orchestrator):
        orchestrator.current_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1", status="blocked")],
            rationale="test"
        )
        orchestrator.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_unknown_status_resets_to_pending_on_new_plan(self, mock_client, mock_settings):
        orchestrator = Orchestrator(goal="test goal", client=mock_client, settings=mock_settings)

        task1 = TaskModel(id="1", description="task 1", status="completed", result="done")
        task2 = TaskModel(id="2", description="task 2", status="blocked", result="stuck")
        orchestrator.current_plan = Plan(tasks=[task1, task2], rationale="initial")

        new_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1"),
                TaskModel(id="2", description="task 2"),
            ],
            rationale="new"
        )

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock, return_value=new_plan):
            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                mock_worker.return_value.run = AsyncMock(return_value="done")
                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester.return_value.run = AsyncMock(
                        return_value=TestResult(passed=True, summary="ok", failures=[])
                    )
                    with patch('furrow.core.orchestrator.console.print'):
                        await orchestrator._cycle()

        tasks = orchestrator.current_plan.tasks
        task2_updated = next(t for t in tasks if t.id == "2")
        assert task2_updated.status == "pending"
        assert task2_updated.result is None