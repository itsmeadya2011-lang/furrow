from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestratorIsDone:
    def test_empty_task_list_is_done(self):
        orchestrator = Orchestrator(goal="test", client=MagicMock())
        orchestrator.tasks = []
        assert orchestrator._is_done() is True

    def test_all_completed_is_done(self):
        orchestrator = Orchestrator(goal="test", client=MagicMock())
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    def test_some_failed_is_not_done(self):
        orchestrator = Orchestrator(goal="test", client=MagicMock())
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ]
        assert orchestrator._is_done() is False

    def test_some_pending_is_not_done(self):
        orchestrator = Orchestrator(goal="test", client=MagicMock())
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ]
        assert orchestrator._is_done() is False


class TestTesterAgentRunTests:
    @pytest.mark.asyncio
    async def test_returns_output_for_first_available_runner(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"tests passed", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_proc

            tester = TesterAgent(client=MagicMock())
            output = await tester._run_tests()

            assert output == "tests passed"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_unavailable_runner_and_tries_next(self):
        mock_proc_fail = AsyncMock()
        mock_proc_fail.communicate = AsyncMock(side_effect=FileNotFoundError("not found"))

        mock_proc_success = AsyncMock()
        mock_proc_success.communicate = AsyncMock(return_value=(b"npm tests passed", b""))
        mock_proc_success.returncode = 0

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [FileNotFoundError("pytest not found"), mock_proc_success]

            tester = TesterAgent(client=MagicMock())
            output = await tester._run_tests()

            assert output == "npm tests passed"
            assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_kills_process_and_continues(self):
        mock_proc_timeout = AsyncMock()
        mock_proc_timeout.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc_timeout.kill = MagicMock()

        mock_proc_success = AsyncMock()
        mock_proc_success.communicate = AsyncMock(return_value=(b"cargo tests passed", b""))
        mock_proc_success.returncode = 0

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [mock_proc_timeout, mock_proc_success]

            tester = TesterAgent(client=MagicMock())
            output = await tester._run_tests()

            assert output == "cargo tests passed"
            mock_proc_timeout.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_no_runner_found_when_all_fail(self):
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = FileNotFoundError("not found")

            tester = TesterAgent(client=MagicMock())
            output = await tester._run_tests()

            assert output == "No test runner found."
            assert mock_create.call_count == 7


class TestWorkerAgentRunWritesFiles:
    @pytest.mark.asyncio
    async def test_write_file_called_with_correct_path(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.complete = AsyncMock(return_value="file content")
        mock_client.write_file = AsyncMock()
        mock_client.settings = Settings(worker_model="test-model")

        task = TaskModel(id="1", description="write file", files=["src/main.py", "src/utils.py"])
        worker = WorkerAgent(task=task, client=mock_client)

        result = await worker.run()

        assert result == "file content"
        assert mock_client.write_file.call_count == 2
        mock_client.write_file.assert_any_call("src/main.py", "file content")
        mock_client.write_file.assert_any_call("src/utils.py", "file content")

    @pytest.mark.asyncio
    async def test_no_write_file_when_no_files_specified(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.complete = AsyncMock(return_value="response")
        mock_client.write_file = AsyncMock()
        mock_client.settings = Settings(worker_model="test-model")

        task = TaskModel(id="1", description="no files", files=[])
        worker = WorkerAgent(task=task, client=mock_client)

        result = await worker.run()

        assert result == "response"
        mock_client.write_file.assert_not_called()


class TestLLMClientMaxTokensFallback:
    @pytest.mark.asyncio
    async def test_complete_uses_settings_max_tokens_when_not_passed(self):
        mock_settings = Settings(
            provider=Provider.ANTHROPIC,
            max_tokens=8192,
            anthropic_api_key="test-key",
        )

        with patch.object(LLMClient, "_complete_anthropic", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "response"

            client = LLMClient(settings=mock_settings)
            result = await client.complete("test prompt")

            assert result == "response"
            mock_complete.assert_called_once()
            call_args = mock_complete.call_args
            assert call_args.kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_complete_uses_explicit_max_tokens_when_passed(self):
        mock_settings = Settings(
            provider=Provider.ANTHROPIC,
            max_tokens=8192,
            anthropic_api_key="test-key",
        )

        with patch.object(LLMClient, "_complete_anthropic", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "response"

            client = LLMClient(settings=mock_settings)
            result = await client.complete("test prompt", max_tokens=2048)

            assert result == "response"
            mock_complete.assert_called_once()
            call_args = mock_complete.call_args
            assert call_args.kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_complete_openai_uses_settings_max_tokens(self):
        mock_settings = Settings(
            provider=Provider.OPENAI,
            max_tokens=4096,
            openai_api_key="test-key",
        )

        with patch.object(LLMClient, "_complete_openai", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "response"

            client = LLMClient(settings=mock_settings)
            result = await client.complete("test prompt")

            assert result == "response"
            mock_complete.assert_called_once()
            call_args = mock_complete.call_args
            assert call_args.kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_complete_ollama_uses_settings_max_tokens(self):
        mock_settings = Settings(
            provider=Provider.OLLAMA,
            max_tokens=2048,
            ollama_base_url="http://localhost:11434",
        )

        with patch.object(LLMClient, "_complete_ollama", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "response"

            client = LLMClient(settings=mock_settings)
            result = await client.complete("test prompt")

            assert result == "response"
            mock_complete.assert_called_once()
            call_args = mock_complete.call_args
            assert call_args.kwargs["max_tokens"] == 2048