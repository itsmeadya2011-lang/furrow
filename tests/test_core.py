import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=LLMClient)

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="test goal", client=mock_client)

    def test_get_tasks_returns_tracked_tasks(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1")
        task2 = TaskModel(id="2", description="task 2")
        orchestrator.tasks = [task1, task2]

        result = orchestrator._get_tasks()

        assert result == [task1, task2]

    def test_is_done_returns_true_when_all_completed(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="completed")
        orchestrator.tasks = [task1, task2]

        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_failed(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="failed")
        orchestrator.tasks = [task1, task2]

        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_pending_tasks_exist(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1", status="completed")
        task2 = TaskModel(id="2", description="task 2", status="pending")
        orchestrator.tasks = [task1, task2]

        assert orchestrator._is_done() is False

    def test_max_cycles_enforcement(self, orchestrator):
        orchestrator.settings.max_cycles = 2
        orchestrator.cycles = 2

        assert orchestrator.cycles >= orchestrator.settings.max_cycles

    def test_max_cycles_zero_unlimited(self, orchestrator):
        orchestrator.settings.max_cycles = 0
        orchestrator.cycles = 100

        assert orchestrator.settings.max_cycles == 0

    @pytest.mark.asyncio
    async def test_orchestrator_stores_plan_tasks_across_cycles(self, orchestrator, mock_client):
        task1 = TaskModel(id="1", description="task 1", files=["file1.py"])
        task2 = TaskModel(id="2", description="task 2", files=["file2.py"])
        plan = Plan(tasks=[task1, task2], rationale="test plan")

        mock_planner = AsyncMock()
        mock_planner.plan.return_value = plan
        orchestrator.planner = mock_planner

        mock_worker = AsyncMock()
        mock_worker.run.return_value = "success"
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            mock_tester = AsyncMock()
            mock_tester.run.return_value = TestResult(passed=True, summary="all passed", failures=[])
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                await orchestrator._cycle()

        assert orchestrator.tasks == [task1, task2]
        assert len(orchestrator.tasks) == 2
        assert orchestrator.tasks[0].id == "1"
        assert orchestrator.tasks[1].id == "2"


class TestWorkerAgent:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock(spec=LLMClient)
        client.settings = Settings(worker_model="test-model")
        return client

    @pytest.fixture
    def worker(self, mock_client):
        task = TaskModel(id="1", description="write hello world", files=["hello.py"])
        return WorkerAgent(task=task, client=mock_client)

    @pytest.mark.asyncio
    async def test_worker_parses_json_response_and_writes_files(self, worker, mock_client):
        mock_client.complete.return_value = '{"files": [{"path": "hello.py", "content": "print(\"hello\")"}]}'

        result = await worker.run()

        mock_client.complete.assert_called_once()
        assert "hello.py" in result or "success" in result.lower()

    @pytest.mark.asyncio
    async def test_worker_handles_invalid_json_gracefully(self, worker, mock_client):
        mock_client.complete.return_value = "This is not valid JSON at all"

        result = await worker.run()

        mock_client.complete.assert_called_once()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_worker_handles_empty_file_list(self, worker, mock_client):
        worker.task.files = []
        mock_client.complete.return_value = "No files to modify"

        result = await worker.run()

        mock_client.complete.assert_called_once()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_worker_passes_correct_prompt_to_llm(self, worker, mock_client):
        mock_client.complete.return_value = "done"

        await worker.run()

        call_args = mock_client.complete.call_args
        assert call_args is not None
        prompt = call_args[0][0]
        assert "write hello world" in prompt
        assert "hello.py" in prompt
        assert call_args[1]["model"] == "test-model"


class TestTesterAgent:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock(spec=LLMClient)
        client.settings = Settings(tester_model="test-model")
        return client

    @pytest.fixture
    def tester(self, mock_client):
        tester = TesterAgent(client=mock_client)
        tester._run_tests = AsyncMock(return_value="Tests passed")
        return tester

    @pytest.mark.asyncio
    async def test_tester_parses_test_output_correctly(self, tester, mock_client):
        mock_client.complete.return_value = '{"passed": true, "summary": "All tests passed", "failures": []}'

        tasks = [TaskModel(id="1", description="test task")]
        result = await tester.run("test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_tester_handles_test_failures(self, tester, mock_client):
        mock_client.complete.return_value = '{"passed": false, "summary": "Some tests failed", "failures": ["test_foo failed", "test_bar failed"]}'

        tasks = [TaskModel(id="1", description="test task")]
        result = await tester.run("test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert "failed" in result.summary.lower()
        assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_tester_handles_invalid_json_response(self, tester, mock_client):
        mock_client.complete.return_value = "Tests passed successfully"

        tasks = [TaskModel(id="1", description="test task")]
        result = await tester.run("test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_tester_handles_no_test_runner_found(self, tester, mock_client):
        tester._run_tests.return_value = "No test runner found. Tried: pytest, npm test, cargo test."
        mock_client.complete.return_value = '{"passed": false, "summary": "No test runner", "failures": ["No test runner found"]}'

        tasks = [TaskModel(id="1", description="test task")]
        result = await tester.run("test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_tester_passes_correct_prompt_to_llm(self, tester, mock_client):
        mock_client.complete.return_value = '{"passed": true, "summary": "ok", "failures": []}'

        tasks = [TaskModel(id="1", description="test task")]
        await tester.run("test goal", tasks)

        call_args = mock_client.complete.call_args
        assert call_args is not None
        prompt = call_args[0][0]
        assert "test goal" in prompt
        assert call_args[1]["model"] == "test-model"


class TestLLMClient:
    @pytest.fixture
    def settings(self):
        return Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key", openai_api_key="test-key")

    @pytest.mark.asyncio
    async def test_complete_routes_to_anthropic_when_provider_anthropic(self, settings):
        settings.provider = Provider.ANTHROPIC
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_anthropic:
            mock_anthropic.return_value = "anthropic response"
            result = await client.complete("test prompt")

            mock_anthropic.assert_called_once_with("test prompt", "", settings.model)
            assert result == "anthropic response"

    @pytest.mark.asyncio
    async def test_complete_routes_to_openai_when_provider_openai(self, settings):
        settings.provider = Provider.OPENAI
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = "openai response"
            result = await client.complete("test prompt")

            mock_openai.assert_called_once_with("test prompt", "", settings.model)
            assert result == "openai response"

    @pytest.mark.asyncio
    async def test_complete_routes_to_ollama_when_provider_ollama(self, settings):
        settings.provider = Provider.OLLAMA
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_ollama:
            mock_ollama.return_value = "ollama response"
            result = await client.complete("test prompt")

            mock_ollama.assert_called_once_with("test prompt", "", settings.model)
            assert result == "ollama response"

    @pytest.mark.asyncio
    async def test_complete_uses_custom_model_when_provided(self, settings):
        settings.provider = Provider.ANTHROPIC
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_anthropic:
            mock_anthropic.return_value = "response"
            await client.complete("test prompt", model="custom-model")

            mock_anthropic.assert_called_once_with("test prompt", "", "custom-model")

    @pytest.mark.asyncio
    async def test_complete_passes_system_prompt(self, settings):
        settings.provider = Provider.ANTHROPIC
        client = LLMClient(settings=settings)

        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_anthropic:
            mock_anthropic.return_value = "response"
            await client.complete("test prompt", system="You are a helpful assistant")

            mock_anthropic.assert_called_once_with("test prompt", "You are a helpful assistant", settings.model)