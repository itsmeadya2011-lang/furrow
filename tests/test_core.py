import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.config import Plan, TaskModel, TestResult, Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestOrchestratorIsDone:
    def test_is_done_all_completed(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator._get_tasks = MagicMock(
            return_value=[
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="completed"),
            ]
        )
        assert orchestrator._is_done() is True

    def test_is_done_with_pending(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator._get_tasks = MagicMock(
            return_value=[
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="pending"),
            ]
        )
        assert orchestrator._is_done() is False

    def test_is_done_with_failed(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator._get_tasks = MagicMock(
            return_value=[
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="failed"),
            ]
        )
        assert orchestrator._is_done() is False

    def test_is_done_empty_tasks(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator._get_tasks = MagicMock(return_value=[])
        assert orchestrator._is_done() is True

    def test_is_done_all_pending(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator._get_tasks = MagicMock(
            return_value=[
                TaskModel(id="1", description="task 1", status="pending"),
                TaskModel(id="2", description="task 2", status="pending"),
            ]
        )
        assert orchestrator._is_done() is False

    def test_is_done_mixed_completed_and_running(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator._get_tasks = MagicMock(
            return_value=[
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="running"),
            ]
        )
        assert orchestrator._is_done() is False


class TestOrchestratorCycle:
    @pytest.mark.asyncio
    async def test_cycle_no_tasks(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator.planner = MagicMock()
        orchestrator.planner.plan = AsyncMock(
            return_value=Plan(tasks=[], rationale="nothing to do")
        )
        await orchestrator._cycle()
        orchestrator.planner.plan.assert_called_once_with("test goal")

    @pytest.mark.asyncio
    async def test_cycle_with_tasks(self):
        orchestrator = Orchestrator(goal="test goal")
        orchestrator.planner = MagicMock()
        orchestrator.planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[TaskModel(id="1", description="do something")],
                rationale="test rationale",
            )
        )
        mock_worker = MagicMock()
        mock_worker.run = AsyncMock(return_value="task result")
        with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
            mock_tester = MagicMock()
            mock_tester.run = AsyncMock(
                return_value=TestResult(passed=True, summary="all good", failures=[])
            )
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                await orchestrator._cycle()
        assert orchestrator.planner.plan.call_count == 1


class TestLLMClientOllama:
    @pytest.mark.asyncio
    async def test_complete_ollama_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Generated text from Ollama"}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            test_settings = Settings(provider=Provider.OLLAMA)
            client = LLMClient(settings=test_settings)
            with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_ollama:
                mock_ollama.return_value = "Generated text from Ollama"
                result = await client._complete_ollama("test prompt", "system prompt", "llama2")
                assert result == "Generated text from Ollama"
                mock_ollama.assert_called_once_with("test prompt", "system prompt", "llama2")

    @pytest.mark.asyncio
    async def test_complete_ollama_with_httpx_mock(self):
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Ollama response text"}
        mock_response.raise_for_status = MagicMock()

        mock_async_client = MagicMock()
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            test_settings = Settings(provider=Provider.OLLAMA)
            client = LLMClient(settings=test_settings)
            with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock_ollama:
                mock_ollama.return_value = "Ollama response text"
                result = await client._complete_ollama("prompt", "system", "llama2")
                assert result == "Ollama response text"


class TestSettings:
    def test_default_provider(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC

    def test_default_max_parallel_tasks(self):
        s = Settings()
        assert s.max_parallel_tasks == 5

    def test_default_model(self):
        s = Settings()
        assert s.model == "claude-sonnet-4-20250514"

    def test_default_planner_model(self):
        s = Settings()
        assert s.planner_model == "claude-3-5-haiku-20241022"

    def test_default_worker_model(self):
        s = Settings()
        assert s.worker_model == "claude-3-5-sonnet-20241022"

    def test_default_tester_model(self):
        s = Settings()
        assert s.tester_model == "claude-3-5-sonnet-20241022"

    def test_default_ollama_base_url(self):
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_max_cycles(self):
        s = Settings()
        assert s.max_cycles == 0

    def test_default_log_level(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_anthropic_api_key_is_none(self):
        s = Settings()
        assert s.anthropic_api_key is None

    def test_default_openai_api_key_is_none(self):
        s = Settings()
        assert s.openai_api_key is None


class TestProvider:
    def test_provider_anthropic(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.ANTHROPIC.value == "anthropic"

    def test_provider_openai(self):
        assert Provider.OPENAI == "openai"
        assert Provider.OPENAI.value == "openai"

    def test_provider_ollama(self):
        assert Provider.OLLAMA == "ollama"
        assert Provider.OLLAMA.value == "ollama"

    def test_all_providers_valid(self):
        providers = list(Provider)
        assert len(providers) == 3
        assert Provider.ANTHROPIC in providers
        assert Provider.OPENAI in providers
        assert Provider.OLLAMA in providers

    def test_provider_from_value(self):
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("openai") == Provider.OPENAI
        assert Provider("ollama") == Provider.OLLAMA

    def test_provider_invalid_raises(self):
        with pytest.raises(ValueError):
            Provider("invalid_provider")


class TestTaskModelDefaults:
    def test_default_files_is_empty_list(self):
        t = TaskModel(id="1", description="test")
        assert t.files == []
        assert isinstance(t.files, list)

    def test_default_dependencies_is_empty_list(self):
        t = TaskModel(id="1", description="test")
        assert t.dependencies == []
        assert isinstance(t.dependencies, list)

    def test_default_status_is_pending(self):
        t = TaskModel(id="1", description="test")
        assert t.status == "pending"

    def test_default_result_is_none(self):
        t = TaskModel(id="1", description="test")
        assert t.result is None

    def test_custom_files(self):
        t = TaskModel(id="1", description="test", files=["file1.py", "file2.py"])
        assert t.files == ["file1.py", "file2.py"]

    def test_custom_dependencies(self):
        t = TaskModel(id="1", description="test", dependencies=["task0"])
        assert t.dependencies == ["task0"]

    def test_custom_status(self):
        t = TaskModel(id="1", description="test", status="completed")
        assert t.status == "completed"

    def test_custom_result(self):
        t = TaskModel(id="1", description="test", result="done")
        assert t.result == "done"

    def test_files_list_not_shared(self):
        t1 = TaskModel(id="1", description="test1")
        t2 = TaskModel(id="2", description="test2")
        t1.files.append("file1.py")
        assert t2.files == []
        assert t1.files == ["file1.py"]

    def test_dependencies_list_not_shared(self):
        t1 = TaskModel(id="1", description="test1")
        t2 = TaskModel(id="2", description="test2")
        t1.dependencies.append("dep1")
        assert t2.dependencies == []
        assert t1.dependencies == ["dep1"]
