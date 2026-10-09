from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestConfig:
    def test_plan_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"

    def test_test_result(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True

    def test_task_model_defaults(self):
        task = TaskModel(id="1", description="test")
        assert task.status == "pending"
        assert task.files == []
        assert task.dependencies == []
        assert task.retry_count == 0
        assert task.max_retries == 3

    def test_settings_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.task_timeout == 300

    def test_settings_validation_max_parallel(self):
        with pytest.raises(ValueError):
            Settings(max_parallel_tasks=0)

    def test_settings_validation_timeout(self):
        with pytest.raises(ValueError):
            Settings(task_timeout=10)


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings()
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="Test goal", client=mock_client)

    def test_orchestrator_init(self, orchestrator):
        assert orchestrator.goal == "Test goal"
        assert orchestrator.cycles == 0
        assert orchestrator.tasks == {}

    def test_is_done_no_tasks(self, orchestrator):
        assert orchestrator._is_done() is True

    def test_is_done_with_completed_tasks(self, orchestrator):
        orchestrator.tasks = {
            "1": TaskModel(id="1", description="task 1", status="completed"),
            "2": TaskModel(id="2", description="task 2", status="completed"),
        }
        assert orchestrator._is_done() is True

    def test_is_done_with_pending_tasks(self, orchestrator):
        orchestrator.tasks = {
            "1": TaskModel(id="1", description="task 1", status="completed"),
            "2": TaskModel(id="2", description="task 2", status="pending"),
        }
        assert orchestrator._is_done() is False

    def test_is_done_with_failed_tasks_max_retries(self, orchestrator):
        orchestrator.tasks = {
            "1": TaskModel(id="1", description="task 1", status="failed", retry_count=3, max_retries=3),
        }
        assert orchestrator._is_done() is True

    def test_is_done_with_failed_tasks_can_retry(self, orchestrator):
        orchestrator.tasks = {
            "1": TaskModel(id="1", description="task 1", status="failed", retry_count=1, max_retries=3),
        }
        assert orchestrator._is_done() is False

    def test_topological_sort_simple(self, orchestrator):
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
            TaskModel(id="3", description="task 3", dependencies=["2"]),
        ]
        sorted_tasks = orchestrator._topological_sort(tasks)
        assert [t.id for t in sorted_tasks] == ["1", "2", "3"]

    def test_topological_sort_parallel(self, orchestrator):
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
            TaskModel(id="3", description="task 3", dependencies=["1", "2"]),
        ]
        sorted_tasks = orchestrator._topological_sort(tasks)
        # 1 and 2 can be in any order, but both before 3
        assert sorted_tasks[2].id == "3"
        assert {t.id for t in sorted_tasks[:2]} == {"1", "2"}

    def test_topological_sort_circular_raises(self, orchestrator):
        tasks = [
            TaskModel(id="1", description="task 1", dependencies=["2"]),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
        ]
        with pytest.raises(ValueError, match="Circular dependency"):
            orchestrator._topological_sort(tasks)


class TestPlannerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings()
        client.complete = AsyncMock(return_value='{"tasks": [{"id": "1", "description": "test task"}], "rationale": "test"}')
        return client

    @pytest.mark.asyncio
    async def test_plan_parses_json(self, mock_client):
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("test goal")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "test task"

    @pytest.mark.asyncio
    async def test_plan_includes_existing_tasks(self, mock_client):
        planner = PlannerAgent(client=mock_client)
        existing = {"1": TaskModel(id="1", description="done task", status="completed")}
        await planner.plan("test goal", existing_tasks=existing)
        assert mock_client.complete.called
        call_args = mock_client.complete.call_args[0][0]
        assert "done task" in call_args
        assert "completed" in call_args


class TestWorkerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings()
        client.complete = AsyncMock(return_value="Task completed successfully")
        return client

    @pytest.mark.asyncio
    async def test_worker_runs(self, mock_client):
        task = TaskModel(id="1", description="test task")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "Task completed successfully"
        mock_client.complete.assert_called_once()


class TestTesterAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings()
        client.complete = AsyncMock(return_value='{"passed": true, "summary": "All tests passed", "failures": []}')
        return client

    @pytest.mark.asyncio
    async def test_tester_passes(self, mock_client):
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, '_run_tests', return_value="test output"):
            result = await tester.run("test goal", [])
        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_tester_fails(self, mock_client):
        mock_client.complete = AsyncMock(return_value='{"passed": false, "summary": "Tests failed", "failures": ["test failed"]}')
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, '_run_tests', return_value="test output"):
            result = await tester.run("test goal", [])
        assert result.passed is False
        assert "test failed" in result.failures

    @pytest.mark.asyncio
    async def test_tester_handles_invalid_json(self, mock_client):
        mock_client.complete = AsyncMock(return_value="not json")
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, '_run_tests', return_value="test output"):
            result = await tester.run("test goal", [])
        assert result.passed is False


class TestLLMClient:
    def test_client_init(self):
        client = LLMClient(Settings(provider=Provider.OPENAI, openai_api_key="test-key"))
        assert client.settings.provider == Provider.OPENAI

    @pytest.mark.asyncio
    async def test_complete_anthropic_mock(self):
        with patch('furrow.llm.AsyncAnthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="test response")]
            mock_anthropic.return_value.messages.create = AsyncMock(return_value=mock_response)

            client = LLMClient(Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key"))
            result = await client.complete("test prompt", model="test-model")

            assert result == "test response"
            mock_anthropic.return_value.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_openai_mock(self):
        with patch('furrow.llm.AsyncOpenAI') as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
            mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            client = LLMClient(Settings(provider=Provider.OPENAI, openai_api_key="test-key"))
            result = await client.complete("test prompt", model="test-model")

            assert result == "test response"
            mock_openai.return_value.chat.completions.create.assert_called_once()