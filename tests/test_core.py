import json
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from tenacity import stop_after_attempt, wait_exponential

from furrow.config import Plan, TaskModel, TestResult, Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient


class TestConfigModels:
    def test_task_model_defaults(self):
        task = TaskModel(id="1", description="do thing")
        assert task.id == "1"
        assert task.description == "do thing"
        assert task.status == "pending"
        assert task.result is None
        assert task.files == []
        assert task.dependencies == []

    def test_task_model_with_all_fields(self):
        task = TaskModel(
            id="2",
            description="complex task",
            files=["file1.py", "file2.py"],
            dependencies=["1"],
            status="completed",
            result="done",
        )
        assert task.files == ["file1.py", "file2.py"]
        assert task.dependencies == ["1"]
        assert task.status == "completed"
        assert task.result == "done"

    def test_plan_with_multiple_tasks(self):
        tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
            TaskModel(id="3", description="task 3", dependencies=["1", "2"]),
        ]
        plan = Plan(tasks=tasks, rationale="sequential tasks")
        assert len(plan.tasks) == 3
        assert plan.tasks[0].id == "1"
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.tasks[2].dependencies == ["1", "2"]

    def test_test_result_with_failures(self):
        result = TestResult(
            passed=False,
            summary="tests failed",
            failures=["test_foo failed", "test_bar failed"],
        )
        assert result.passed is False
        assert result.summary == "tests failed"
        assert result.failures == ["test_foo failed", "test_bar failed"]

    def test_test_result_empty_failures_default(self):
        result = TestResult(passed=True, summary="all good")
        assert result.failures == []

    def test_provider_enum_values(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"
        assert list(Provider) == [Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA]


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.max_parallel_tasks = 2
        client.settings.max_cycles = 0
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="test goal", client=mock_client)

    @pytest.mark.asyncio
    async def test_is_done_all_completed_no_failures(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_with_failed_task(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed", result="error"),
        ]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_with_pending_task(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_empty_tasks(self, orchestrator):
        orchestrator.tasks = []
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_respects_max_cycles(self, orchestrator):
        with patch("furrow.core.orchestrator.settings") as mock_settings:
            mock_settings.max_cycles = 3
            orchestrator.cycles = 3
            orchestrator.tasks = [TaskModel(id="1", description="t1", status="pending")]
            assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_task_merging_deduplication_by_id(self, orchestrator):
        orchestrator.tasks = [TaskModel(id="1", description="original")]
        
        new_plan = Plan(
            tasks=[
                TaskModel(id="1", description="duplicate"),
                TaskModel(id="2", description="new task"),
            ],
            rationale="test",
        )
        
        existing_ids = {t.id for t in orchestrator.tasks}
        for task in new_plan.tasks:
            if task.id not in existing_ids:
                orchestrator.tasks.append(task)
                existing_ids.add(task.id)
        
        assert len(orchestrator.tasks) == 2
        assert orchestrator.tasks[0].description == "original"
        assert orchestrator.tasks[1].id == "2"

    @pytest.mark.asyncio
    async def test_max_parallel_tasks_limits_concurrency(self):
        with patch("furrow.core.orchestrator.settings") as mock_settings:
            mock_settings.max_parallel_tasks = 2
            mock_settings.max_cycles = 0
            
            with patch("furrow.core.orchestrator.asyncio.Semaphore") as mock_semaphore_class:
                mock_semaphore = AsyncMock()
                mock_semaphore_class.return_value = mock_semaphore
                
                # Mock WorkerAgent to return completed immediately
                with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
                    mock_worker = AsyncMock()
                    mock_worker.run = AsyncMock(return_value="result")
                    mock_worker_class.return_value = mock_worker
                    
                    mock_client = AsyncMock(spec=LLMClient)
                    mock_client.settings = MagicMock()
                    mock_client.settings.max_parallel_tasks = 2
                    mock_client.settings.max_cycles = 0
                    
                    orchestrator = Orchestrator(goal="test", client=mock_client)
                    orchestrator.tasks = [
                        TaskModel(id="1", description="t1", status="pending"),
                        TaskModel(id="2", description="t2", status="pending"),
                    ]
                    orchestrator.planner.plan = AsyncMock(return_value=Plan(
                        tasks=orchestrator.tasks, rationale="test"
                    ))
                    
                    with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
                        mock_tester = AsyncMock()
                        mock_tester.run = AsyncMock(return_value=TestResult(
                            passed=True, summary="all good", failures=[]
                        ))
                        mock_tester_class.return_value = mock_tester
                        
                        await orchestrator._cycle()
                
                # Verify semaphore was created with max_parallel_tasks from settings
                mock_semaphore_class.assert_called_once_with(2)


import asyncio


class TestWorkerAgent:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.worker_model = "test-model"
        client.settings.workspace = Path("/workspace")
        client.read_file = AsyncMock(return_value="file content")
        client.write_file = AsyncMock()
        return client

    @pytest.fixture
    def worker(self, mock_client):
        task = TaskModel(
            id="1",
            description="read and write a file",
            files=["test.py"],
        )
        return WorkerAgent(task=task, client=mock_client)

    @pytest.mark.asyncio
    async def test_worker_parses_json_actions_and_executes(self, worker, mock_client):
        mock_client.complete = AsyncMock(side_effect=[
            json.dumps({
                "actions": [
                    {"type": "read", "path": "test.py"},
                    {"type": "write", "path": "test.py", "content": "new content"},
                ],
                "summary": "done"
            }),
            json.dumps({"actions": [], "summary": "completed"})
        ])
        
        result = await worker.run()
        
        assert mock_client.complete.call_count == 2
        assert mock_client.read_file.called
        assert mock_client.write_file.called
        assert "completed" in result

    @pytest.mark.asyncio
    async def test_worker_respects_allowed_files(self, worker, mock_client):
        mock_client.complete = AsyncMock(side_effect=[
            json.dumps({
                "actions": [
                    {"type": "read", "path": "test.py"},
                    {"type": "read", "path": "forbidden.py"},
                ],
                "summary": "round 1"
            }),
            json.dumps({
                "actions": [],
                "summary": "skipped: forbidden.py is outside allowed files"
            })
        ])
        
        result = await worker.run()
        
        assert "skipped: forbidden.py is outside allowed files" in result
        mock_client.read_file.assert_called_once_with("test.py")

    @pytest.mark.asyncio
    async def test_worker_loops_up_to_3_rounds(self, worker, mock_client):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "actions": [{"type": "bash", "command": "echo hello"}],
            "summary": "round done"
        }))
        
        result = await worker.run()
        
        assert mock_client.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_worker_returns_summary_when_no_actions(self, worker, mock_client):
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "actions": [],
            "summary": "nothing to do"
        }))
        
        result = await worker.run()
        
        assert result == "nothing to do"
        assert mock_client.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_worker_handles_json_decode_error(self, worker, mock_client):
        mock_client.complete = AsyncMock(return_value="not valid json")
        
        result = await worker.run()
        
        assert "Failed to parse LLM JSON response" in result
        assert "not valid json" in result

    @pytest.mark.asyncio
    async def test_worker_handles_unknown_action_type(self, worker, mock_client):
        mock_client.complete = AsyncMock(side_effect=[
            json.dumps({
                "actions": [{"type": "unknown", "path": "test.py"}],
                "summary": "round 1"
            }),
            json.dumps({
                "actions": [],
                "summary": "unknown_action_type: unknown"
            })
        ])
        
        result = await worker.run()
        
        assert "unknown_action_type: unknown" in result


class TestLLMClient:
    @pytest.fixture
    def settings(self):
        return Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_ollama_calls_correct_endpoint(self, settings):
        with patch("furrow.llm.httpx.AsyncClient") as mock_client_class:
            mock_response = AsyncMock()
            mock_response.json = MagicMock(return_value={"response": "ollama output"})
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            client = LLMClient(settings=settings)
            client._complete_ollama = AsyncMock(return_value="ollama output")
            result = await client.complete("test prompt", model="test-model", timeout=30.0)
            
            assert result == "ollama output"
            client._complete_ollama.assert_called_once_with("test prompt", "", "test-model", 30.0)

    @pytest.mark.asyncio
    async def test_retry_logic_works(self, settings):
        """Test that _complete_ollama has tenacity retry decorator configured."""
        client = LLMClient(settings=settings)
        
        # Verify the method has tenacity retry wrapper
        assert hasattr(client._complete_ollama, 'retry')
        retry = client._complete_ollama.retry
        # Check retry configuration - stop after 3 attempts
        assert retry.stop.__class__.__name__ == 'stop_after_attempt'
        assert retry.stop.max_attempt_number == 3
        # Check wait configuration - exponential backoff
        assert retry.wait.__class__.__name__ == 'wait_exponential'
        assert retry.wait.multiplier == 1
        assert retry.wait.min == 1
        assert retry.wait.max == 10

    @pytest.mark.asyncio
    async def test_timeout_passed_through(self, settings):
        captured_timeout = {}
        
        async def mock_complete_ollama(prompt, system, model, timeout):
            captured_timeout["timeout"] = timeout
            return "output"
        
        client = LLMClient(settings=settings)
        client._complete_ollama = mock_complete_ollama
        await client.complete("test prompt", model="test-model", timeout=15.0)
        
        assert captured_timeout["timeout"] == 15.0

    @pytest.mark.asyncio
    async def test_anthropic_provider_uses_correct_client(self, settings):
        settings.provider = Provider.ANTHROPIC
        settings.anthropic_api_key = "test-key"
        
        client = LLMClient(settings=settings)
        with patch("furrow.llm.AsyncAnthropic") as mock_anthropic:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=MagicMock(
                content=[MagicMock(text="anthropic output")]
            ))
            mock_anthropic.return_value = mock_instance
            
            client._complete_anthropic = AsyncMock(return_value="anthropic output")
            result = await client.complete("test prompt", model="test-model")
            
            assert result == "anthropic output"

    @pytest.mark.asyncio
    async def test_openai_provider_uses_correct_client(self, settings):
        settings.provider = Provider.OPENAI
        settings.openai_api_key = "test-key"
        
        client = LLMClient(settings=settings)
        client._complete_openai = AsyncMock(return_value="openai output")
        result = await client.complete("test prompt", model="test-model")
        
        assert result == "openai output"

    @pytest.mark.asyncio
    async def test_read_file(self, settings):
        client = LLMClient(settings=settings)
        with patch("furrow.llm.aiofiles.open") as mock_open:
            mock_file = AsyncMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock()
            mock_file.read = AsyncMock(return_value="file content")
            mock_open.return_value = mock_file
            
            result = await client.read_file("test.py")
            
            assert result == "file content"

    @pytest.mark.asyncio
    async def test_write_file(self, settings):
        client = LLMClient(settings=settings)
        with patch("furrow.llm.aiofiles.open") as mock_open, \
             patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_file = AsyncMock()
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock()
            mock_file.write = AsyncMock()
            mock_open.return_value = mock_file
            
            await client.write_file("test.py", "content")
            
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_file.write.assert_called_once_with("content")

    @pytest.mark.asyncio
    async def test_list_files(self, settings):
        client = LLMClient(settings=settings)
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.rglob") as mock_rglob:
            mock_file1 = MagicMock()
            mock_file1.is_file.return_value = True
            mock_file1.relative_to = MagicMock(return_value=Path("file1.py"))
            
            mock_file2 = MagicMock()
            mock_file2.is_file.return_value = True
            mock_file2.relative_to = MagicMock(return_value=Path("dir/file2.py"))
            
            mock_dir = MagicMock()
            mock_dir.is_file.return_value = False
            mock_dir.relative_to = MagicMock(return_value=Path("dir"))
            
            mock_rglob.return_value = [mock_file1, mock_dir, mock_file2]
            
            result = client.list_files(".")
            
            assert "file1.py" in result
            assert "dir/file2.py" in result
            assert "dir" not in result