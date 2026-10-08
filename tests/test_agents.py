from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class FakeLLMClient:
    def __init__(
        self,
        response: str = "{}",
        files: dict[str, str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.response = response
        self.files = files or {}
        self.write_calls: list[tuple[str, str]] = []
        self.complete_calls: list[tuple[str, str | None]] = []
        self._settings = settings or Settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append((prompt, model))
        return self.response

    async def read_file(self, path: str | Path) -> str:
        return self.files.get(str(path), "")

    async def write_file(self, path: str | Path, content: str) -> None:
        self.write_calls.append((str(path), content))

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_instantiation_with_task_model(self) -> None:
        task = TaskModel(id="1", description="Test task", files=["test.py"])
        client = FakeLLMClient()
        agent = WorkerAgent(task=task, client=client)
        assert agent.task == task
        assert agent.client is client

    @pytest.mark.asyncio
    async def test_strip_json_markers_removes_code_fences(self) -> None:
        client = FakeLLMClient()
        agent = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)

        assert agent._strip_json_markers('{"key": "value"}') == '{"key": "value"}'
        assert agent._strip_json_markers('```json\n{"key": "value"}\n```') == '{"key": "value"}'
        assert agent._strip_json_markers('```\n{"key": "value"}\n```') == '{"key": "value"}'
        assert agent._strip_json_markers('```JSON\n{"key": "value"}\n```') == '{"key": "value"}'
        assert agent._strip_json_markers('  ```json\n{"key": "value"}\n```  ') == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_run_parses_json_response_and_writes_files(self, tmp_path: Path) -> None:
        task = TaskModel(id="1", description="Create a file", files=["output.txt"])
        response_data = {
            "files": {"output.txt": "Hello, World!"},
            "summary": "Created output.txt",
        }
        client = FakeLLMClient(response=json.dumps(response_data), files={})
        agent = WorkerAgent(task=task, client=client)

        result = await agent.run()

        assert result == "Created output.txt"
        assert len(client.write_calls) == 1
        assert client.write_calls[0][0] == "output.txt"
        assert client.write_calls[0][1] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json_response(self) -> None:
        task = TaskModel(id="1", description="Test task", files=["test.py"])
        client = FakeLLMClient(response="not valid json")
        agent = WorkerAgent(task=task, client=client)

        result = await agent.run()

        assert "Failed to parse worker response" in result

    @pytest.mark.asyncio
    async def test_run_reads_existing_files(self, tmp_path: Path) -> None:
        existing_file = tmp_path / "existing.py"
        existing_file.write_text("print('hello')")

        task = TaskModel(id="1", description="Modify file", files=[str(existing_file)])
        response_data = {"files": {str(existing_file): "print('world')"}, "summary": "Modified"}
        client = FakeLLMClient(response=json.dumps(response_data), files={str(existing_file): "print('hello')"})
        agent = WorkerAgent(task=task, client=client)

        result = await agent.run()

        assert result == "Modified"
        assert client.write_calls[0][1] == "print('world')"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_instantiation(self) -> None:
        client = FakeLLMClient()
        agent = TesterAgent(client=client)
        assert agent.client is client

    @pytest.mark.asyncio
    async def test_strip_json_markers_removes_code_fences(self) -> None:
        client = FakeLLMClient()
        agent = TesterAgent(client=client)

        assert agent._strip_json_markers('{"passed": true}') == '{"passed": true}'
        assert agent._strip_json_markers('```json\n{"passed": true}\n```') == '{"passed": true}'
        assert agent._strip_json_markers('```\n{"passed": true}\n```') == '{"passed": true}'

    @pytest.mark.asyncio
    async def test_run_parses_passed_response(self) -> None:
        response_data = {
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        }
        client = FakeLLMClient(response=json.dumps(response_data))
        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task 1")]

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="test output"):
            result = await agent.run("Test goal", tasks)

        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_parses_failed_response_with_fixes(self, tmp_path: Path) -> None:
        fix_file = tmp_path / "fix_me.py"
        fix_file.write_text("broken")

        response_data = {
            "passed": False,
            "summary": "Tests failed",
            "failures": ["AssertionError: expected 1, got 0"],
            "fixes": [{"path": str(fix_file), "content": "fixed"}],
        }
        client = FakeLLMClient(response=json.dumps(response_data))
        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task 1")]

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="test output"):
            result = await agent.run("Test goal", tasks)

        assert result.passed is False
        assert len(client.write_calls) == 1
        assert client.write_calls[0][0] == str(fix_file)
        assert client.write_calls[0][1] == "fixed"

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json(self) -> None:
        client = FakeLLMClient(response="not valid json")
        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="Task 1")]

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="test output"):
            result = await agent.run("Test goal", tasks)

        assert result.passed is False
        assert "not valid json" in result.summary


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_instantiation(self) -> None:
        client = FakeLLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        assert orchestrator.goal == "Test goal"
        assert orchestrator.client is client
        assert orchestrator.tasks == []

    @pytest.mark.asyncio
    async def test_get_tasks_returns_stored_tasks(self) -> None:
        client = FakeLLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        tasks = [
            TaskModel(id="1", description="Task 1"),
            TaskModel(id="2", description="Task 2"),
        ]
        orchestrator.tasks = tasks

        result = orchestrator._get_tasks()

        assert result == tasks
        assert result is orchestrator.tasks  # Returns the same list reference

    @pytest.mark.asyncio
    async def test_is_done_returns_false_when_no_tasks(self) -> None:
        client = FakeLLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_returns_false_when_tasks_failed(self) -> None:
        client = FakeLLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        orchestrator.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed"),
        ]

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_returns_true_when_all_completed(self) -> None:
        client = FakeLLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        orchestrator.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_returns_false_when_pending_tasks(self) -> None:
        client = FakeLLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        orchestrator.tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="pending"),
        ]

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_stores_tasks_from_plan(self) -> None:
        plan_tasks = [
            TaskModel(id="1", description="Task 1"),
            TaskModel(id="2", description="Task 2"),
        ]
        plan = Plan(tasks=plan_tasks, rationale="Test plan")
        client = FakeLLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)

        orchestrator.tasks = plan.tasks

        assert orchestrator.tasks == plan_tasks


class TestConfigModels:
    def test_task_model_creation(self) -> None:
        task = TaskModel(id="1", description="Test task", files=["a.py", "b.py"])
        assert task.id == "1"
        assert task.description == "Test task"
        assert task.files == ["a.py", "b.py"]
        assert task.dependencies == []
        assert task.status == "pending"
        assert task.result is None

    def test_task_model_with_all_fields(self) -> None:
        task = TaskModel(
            id="2",
            description="Complex task",
            files=["main.py"],
            dependencies=["1"],
            status="completed",
            result="Done",
        )
        assert task.dependencies == ["1"]
        assert task.status == "completed"
        assert task.result == "Done"

    def test_plan_creation(self) -> None:
        tasks = [TaskModel(id="1", description="Task 1")]
        plan = Plan(tasks=tasks, rationale="Test rationale")
        assert plan.tasks == tasks
        assert plan.rationale == "Test rationale"

    def test_test_result_passed(self) -> None:
        result = TestResult(passed=True, summary="All good", failures=[])
        assert result.passed is True
        assert result.summary == "All good"
        assert result.failures == []

    def test_test_result_failed(self) -> None:
        result = TestResult(passed=False, summary="Failed", failures=["Error 1", "Error 2"])
        assert result.passed is False
        assert result.summary == "Failed"
        assert result.failures == ["Error 1", "Error 2"]

    def test_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.provider.value == "anthropic"
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.max_parallel_tasks == 5
        assert settings.max_cycles == 0


class TestFakeLLMClient:
    @pytest.mark.asyncio
    async def test_complete_records_calls(self) -> None:
        client = FakeLLMClient(response="test response")
        result = await client.complete("prompt", model="test-model")

        assert result == "test response"
        assert len(client.complete_calls) == 1
        assert client.complete_calls[0][0] == "prompt"
        assert client.complete_calls[0][1] == "test-model"

    @pytest.mark.asyncio
    async def test_read_file_returns_stored_content(self) -> None:
        client = FakeLLMClient(files={"test.py": "print('hello')"})
        content = await client.read_file("test.py")

        assert content == "print('hello')"

    @pytest.mark.asyncio
    async def test_read_file_returns_empty_for_missing(self) -> None:
        client = FakeLLMClient()
        content = await client.read_file("missing.py")

        assert content == ""

    @pytest.mark.asyncio
    async def test_write_file_records_calls(self, tmp_path: Path) -> None:
        client = FakeLLMClient()
        await client.write_file("output.txt", "content")

        assert len(client.write_calls) == 1
        assert client.write_calls[0] == ("output.txt", "content")