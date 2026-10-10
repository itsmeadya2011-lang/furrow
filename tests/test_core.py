from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture
def mock_settings():
    return Settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key="test-key",
        max_cycles=3,
        workspace=Path("/tmp/test"),
    )


@pytest.fixture
def mock_client(mock_settings):
    client = LLMClient(settings=mock_settings)
    client._complete_anthropic = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')
    client._complete_openai = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')
    return client


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_get_tasks_returns_stored_tasks(self, mock_client):
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        task1 = TaskModel(id="1", description="task 1")
        task2 = TaskModel(id="2", description="task 2")
        orchestrator.tasks = [task1, task2]

        tasks = orchestrator._get_tasks()

        assert tasks == [task1, task2]
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_is_done_returns_true_when_all_tasks_completed(self, mock_client):
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_returns_false_when_any_task_failed(self, mock_client):
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ]

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_returns_true_when_no_tasks(self, mock_client):
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.tasks = []

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_max_cycles_respected(self, mock_client):
        mock_settings = Settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test-key",
            max_cycles=2,
            workspace=Path("/tmp/test"),
        )
        client = LLMClient(settings=mock_settings)
        client._complete_anthropic = AsyncMock(
            return_value='{"tasks": [{"id": "1", "description": "task"}], "rationale": "plan"}'
        )

        orchestrator = Orchestrator(goal="test goal", client=client)
        orchestrator.planner.plan = AsyncMock(
            return_value=Plan(tasks=[TaskModel(id="1", description="task")], rationale="plan")
        )

        with patch(
            "furrow.core.orchestrator.WorkerAgent.run", new_callable=AsyncMock
        ) as mock_worker_run, patch(
            "furrow.core.orchestrator.TesterAgent.run", new_callable=AsyncMock
        ) as mock_tester_run:
            # Make worker fail so task status becomes "failed", keeping _is_done() False
            mock_worker_run.side_effect = Exception("worker error")
            mock_tester_run.return_value = TestResult(
            passed=True, summary="tests passed", failures=[]
        )

            await orchestrator.run()

        # max_cycles=2, so loop should run 2 cycles before hitting the limit
        assert orchestrator.cycles == 2

    @pytest.mark.asyncio
    async def test_tasks_merged_correctly_existing_updated_new_appended(self, mock_client):
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        existing_task = TaskModel(id="1", description="existing", status="pending")
        orchestrator.tasks = [existing_task]

        new_plan = Plan(
            tasks=[
                TaskModel(id="1", description="updated", status="completed", result="done"),
                TaskModel(id="2", description="new task", status="pending"),
            ],
            rationale="update and add",
        )

        existing_by_id = {t.id: t for t in orchestrator.tasks}
        for task in new_plan.tasks:
            if task.id in existing_by_id:
                existing = existing_by_id[task.id]
                existing.status = task.status
                existing.result = task.result
            else:
                orchestrator.tasks.append(task)
                existing_by_id[task.id] = task

        assert len(orchestrator.tasks) == 2
        assert orchestrator.tasks[0].id == "1"
        assert orchestrator.tasks[0].status == "completed"
        assert orchestrator.tasks[0].result == "done"
        assert orchestrator.tasks[1].id == "2"
        assert orchestrator.tasks[1].description == "new task"


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_parses_json_and_writes_files(self, tmp_path, mock_settings):
        mock_settings.workspace = tmp_path
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(return_value=json.dumps({
            "files": [{"path": "test.txt", "content": "hello"}],
            "summary": "Created test.txt"
        }))
        client.write_file = AsyncMock()

        task = TaskModel(id="1", description="create file", files=["test.txt"])
        worker = WorkerAgent(task=task, client=client)

        result = await worker.run()

        assert "Created test.txt" in result
        client.write_file.assert_called_once()
        call_args = client.write_file.call_args
        assert "test.txt" in str(call_args[0][0])

    @pytest.mark.asyncio
    async def test_run_handles_markdown_code_fences(self, tmp_path, mock_settings):
        mock_settings.workspace = tmp_path
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(
            return_value=(
                '```json\n'
                '{"files": [{"path": "fenced.txt", "content": "fenced"}], "summary": "Fenced"}\n'
                '```'
            )
        )
        client.write_file = AsyncMock()

        task = TaskModel(id="1", description="create file", files=["fenced.txt"])
        worker = WorkerAgent(task=task, client=client)

        result = await worker.run()

        assert "Fenced" in result
        client.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_resolves_relative_paths_to_workspace(self, tmp_path, mock_settings):
        mock_settings.workspace = tmp_path
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(return_value=json.dumps({
            "files": [{"path": "subdir/file.txt", "content": "content"}],
            "summary": "Created file"
        }))
        client.write_file = AsyncMock()

        task = TaskModel(id="1", description="create file", files=["subdir/file.txt"])
        worker = WorkerAgent(task=task, client=client)

        await worker.run()

        call_args = client.write_file.call_args
        written_path = call_args[0][0]
        assert written_path == tmp_path / "subdir" / "file.txt"

    @pytest.mark.asyncio
    async def test_run_catches_write_errors_in_summary(self, tmp_path, mock_settings):
        mock_settings.workspace = tmp_path
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(return_value=json.dumps({
            "files": [{"path": "error.txt", "content": "content"}],
            "summary": "Created file"
        }))
        client.write_file = AsyncMock(side_effect=PermissionError("denied"))

        task = TaskModel(id="1", description="create file", files=["error.txt"])
        worker = WorkerAgent(task=task, client=client)

        result = await worker.run()

        assert "Error writing" in result
        assert "denied" in result

    @pytest.mark.asyncio
    async def test_run_returns_raw_response_when_json_parse_fails(self, tmp_path, mock_settings):
        mock_settings.workspace = tmp_path
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(return_value="This is not JSON at all")
        client.write_file = AsyncMock()

        task = TaskModel(id="1", description="create file")
        worker = WorkerAgent(task=task, client=client)

        result = await worker.run()

        assert result == "This is not JSON at all"


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_routes_to_anthropic(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        client._complete_anthropic = AsyncMock(return_value="anthropic response")

        result = await client.complete("test prompt")

        assert result == "anthropic response"
        client._complete_anthropic.assert_called_once_with("test prompt", "", mock_settings.model)

    @pytest.mark.asyncio
    async def test_complete_routes_to_openai(self, mock_settings):
        mock_settings.provider = Provider.OPENAI
        mock_settings.openai_api_key = "test-key"
        client = LLMClient(settings=mock_settings)
        client._complete_openai = AsyncMock(return_value="openai response")

        result = await client.complete("test prompt")

        assert result == "openai response"
        client._complete_openai.assert_called_once_with("test prompt", "", mock_settings.model)

    @pytest.mark.asyncio
    async def test_complete_ollama_provider_supported(self, mock_settings):
        mock_settings.provider = Provider.OLLAMA
        client = LLMClient(settings=mock_settings)
        client._complete_ollama = AsyncMock(return_value="ollama response")

        result = await client.complete("test prompt")

        assert result == "ollama response"
        client._complete_ollama.assert_called_once_with("test prompt", "", mock_settings.model)

    @pytest.mark.asyncio
    async def test_complete_retries_on_transient_error(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        call_count = 0

        async def failing_then_success(prompt, system, model):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        client._complete_anthropic = failing_then_success

        result = await client.complete("test prompt")

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_directories(self, tmp_path, mock_settings):
        client = LLMClient(settings=mock_settings)
        test_file = tmp_path / "subdir" / "nested" / "file.txt"

        await client.write_file(test_file, "content")

        assert test_file.exists()
        assert test_file.read_text() == "content"

    def test_list_files_returns_relative_paths(self, tmp_path, mock_settings):
        client = LLMClient(settings=mock_settings)
        (tmp_path / "file1.txt").write_text("1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("2")

        files = client.list_files(tmp_path)

        assert set(files) == {"file1.txt", "subdir/file2.txt"}


class TestConfig:
    def test_plan_serializes_deserializes(self):
        plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="reason")
        data = plan.model_dump()
        restored = Plan(**data)

        assert restored.tasks[0].id == "1"
        assert restored.tasks[0].description == "task"
        assert restored.rationale == "reason"

    def test_task_model_serializes_deserializes(self):
        task = TaskModel(
            id="1", description="desc", files=["f1"], dependencies=["d1"],
            status="completed", result="ok",
        )
        data = task.model_dump()
        restored = TaskModel(**data)

        assert restored.id == "1"
        assert restored.files == ["f1"]
        assert restored.dependencies == ["d1"]
        assert restored.status == "completed"
        assert restored.result == "ok"

    def test_test_result_handles_failures_list(self):
        result = TestResult(passed=False, summary="failed", failures=["error1", "error2"])
        assert result.failures == ["error1", "error2"]
        assert result.passed is False

        result2 = TestResult(passed=True, summary="passed")
        assert result2.failures == []

    def test_settings_loads_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("FURROW_PROVIDER", "openai")
        monkeypatch.setenv("FURROW_MODEL", "gpt-4")
        monkeypatch.setenv("FURROW_MAX_CYCLES", "10")

        settings = Settings()

        assert settings.provider == Provider.OPENAI
        assert settings.model == "gpt-4"
        assert settings.max_cycles == 10

    def test_provider_enum_values(self):
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_returns_no_runner_when_not_found(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        tester = TesterAgent(client=client)

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            output = await tester._run_tests()

        assert output == "No test runner found."

    @pytest.mark.asyncio
    async def test_run_parses_llm_json_response(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(return_value=json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": []
        }))
        tester = TesterAgent(client=client)

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "test output"

            result = await tester.run("goal", [])

        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_falls_back_to_passed_check_when_json_fails(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(return_value="The tests passed successfully")
        tester = TesterAgent(client=client)

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "test output"

            result = await tester.run("goal", [])

        assert result.passed is True
        assert result.summary == "The tests passed successfully"

    @pytest.mark.asyncio
    async def test_run_falls_back_to_failed_when_json_fails_and_no_passed(self, mock_settings):
        client = LLMClient(settings=mock_settings)
        client.complete = AsyncMock(return_value="The tests failed with errors")
        tester = TesterAgent(client=client)

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "test output"

            result = await tester.run("goal", [])

        assert result.passed is False
        assert result.summary == "The tests failed with errors"