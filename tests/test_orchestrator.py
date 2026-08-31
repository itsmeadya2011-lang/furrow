from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestratorIsDone:
    def test_empty_tasks_returns_true(self) -> None:
        orchestrator = Orchestrator(goal="test")
        assert orchestrator._is_done() is True

    def test_all_completed_returns_true(self) -> None:
        orchestrator = Orchestrator(goal="test")
        orchestrator.tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert orchestrator._is_done() is True

    def test_any_failed_returns_false(self) -> None:
        orchestrator = Orchestrator(goal="test")
        orchestrator.tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ]
        assert orchestrator._is_done() is False

    def test_any_pending_returns_false(self) -> None:
        orchestrator = Orchestrator(goal="test")
        orchestrator.tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ]
        assert orchestrator._is_done() is False


class TestOrchestratorGetTasks:
    def test_get_tasks_returns_stored_tasks(self) -> None:
        orchestrator = Orchestrator(goal="test")
        tasks = [TaskModel(id="1", description="a")]
        orchestrator.tasks = tasks
        assert orchestrator._get_tasks() == tasks


class TestOrchestratorMaxCycles:
    @pytest.mark.asyncio
    async def test_max_cycles_enforced(self) -> None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = Settings(max_cycles=2)
        mock_planner = AsyncMock()
        # Always return a task so _is_done never short-circuits
        mock_planner.plan.return_value = Plan(
            tasks=[TaskModel(id="1", description="never done")],
            rationale="loop",
        )
        mock_tester_instance = MagicMock()
        mock_tester_instance.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["x"])
        )
        
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.goal = "test"
        orchestrator.client = mock_client
        orchestrator.planner = mock_planner
        orchestrator.cycles = 0
        orchestrator.tasks = []
        
        with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockTester.return_value = mock_tester_instance
            await orchestrator.run()
        
        assert orchestrator.cycles == 2


class TestOllamaCompletion:
    @pytest.mark.asyncio
    async def test_complete_ollama(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "ollama response"}}
        mock_response.raise_for_status = MagicMock()
        
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        
        with patch("furrow.llm.httpx.AsyncClient", return_value=mock_client):
            settings = Settings(provider="ollama", model="llama3", ollama_base_url="http://localhost:11434")
            llm = LLMClient(settings=settings)
            result = await llm.complete("hello", model="llama3")
            assert result == "ollama response"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/chat"
            payload = call_args[1]["json"]
            assert payload["model"] == "llama3"
            assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_complete_ollama_http_error(self) -> None:
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "server error", request=MagicMock(), response=mock_response
            )
        )
        
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        
        with patch("furrow.llm.httpx.AsyncClient", return_value=mock_client):
            settings = Settings(provider="ollama", model="llama3", ollama_base_url="http://localhost:11434")
            llm = LLMClient(settings=settings)
            with pytest.raises(ValueError, match="Ollama request failed"):
                await llm.complete("hello", model="llama3")
