from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class MockLLMClient(LLMClient):
    def __init__(
        self, responses: dict[str, str] | None = None, settings: Settings | None = None
    ) -> None:
        super().__init__(settings=settings or Settings())
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if (
            "PLANNER_PROMPT" in prompt
            or "planning agent" in prompt.lower()
            or "break the user's goal" in prompt.lower()
        ):
            return self.responses.get(
                "planner", self.responses.get("default", '{"tasks": [], "rationale": "Done"}')
            )
        if (
            "WORKER_PROMPT" in prompt
            or "worker agent" in prompt.lower()
            or "implement the assigned task" in prompt.lower()
        ):
            return self.responses.get(
                "worker", self.responses.get("default", '{"files": [], "summary": "Done"}')
            )
        if (
            "TESTER_PROMPT" in prompt
            or "tester agent" in prompt.lower()
            or "determine if tests passed" in prompt.lower()
        ):
            return self.responses.get(
                "tester",
                self.responses.get(
                    "default", '{"passed": true, "summary": "All tests passed", "failures": []}'
                ),
            )
        return self.responses.get("default", '{"files": [], "summary": "Done"}')


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(max_cycles=3)


@pytest.mark.asyncio
async def test_orchestrator_task_tracking(mock_settings: Settings) -> None:
    client = MockLLMClient(
        {
            "planner": '{"tasks": [{"id": "1", "description": "Task 1", "files": ["file1.py"], "dependencies": []}, {"id": "2", "description": "Task 2", "files": ["file2.py"], "dependencies": []}], "rationale": "Test plan"}',
            "worker": '{"files": [{"path": "file1.py", "content": "print(1)"}], "summary": "Created file1"}',
            "tester": '{"passed": true, "summary": "All tests passed", "failures": []}',
        },
        mock_settings,
    )

    orchestrator = Orchestrator(goal="Test goal", client=client, settings=mock_settings)

    with patch("furrow.agents.tester.TesterAgent.run", new_callable=AsyncMock) as mock_test:
        mock_test.return_value = TestResult(passed=True, summary="All tests passed", failures=[])
        await orchestrator._cycle()

    assert len(orchestrator.tasks) == 2
    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.tasks[1].status == "completed"
    assert orchestrator._get_tasks() == orchestrator.tasks


@pytest.mark.asyncio
async def test_orchestrator_is_done_all_completed(mock_settings: Settings) -> None:
    client = MockLLMClient({}, mock_settings)
    orchestrator = Orchestrator(goal="Test goal", client=client, settings=mock_settings)

    task1 = TaskModel(id="1", description="Task 1", status="completed")
    task2 = TaskModel(id="2", description="Task 2", status="completed")
    orchestrator.tasks = [task1, task2]

    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failed(mock_settings: Settings) -> None:
    client = MockLLMClient({}, mock_settings)
    orchestrator = Orchestrator(goal="Test goal", client=client, settings=mock_settings)

    task1 = TaskModel(id="1", description="Task 1", status="completed")
    task2 = TaskModel(id="2", description="Task 2", status="failed")
    orchestrator.tasks = [task1, task2]

    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_empty_tasks(mock_settings: Settings) -> None:
    client = MockLLMClient({}, mock_settings)
    orchestrator = Orchestrator(goal="Test goal", client=client, settings=mock_settings)
    orchestrator.tasks = []

    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_max_cycles(mock_settings: Settings) -> None:
    mock_settings.max_cycles = 2
    client = MockLLMClient({}, mock_settings)
    orchestrator = Orchestrator(goal="Test goal", client=client, settings=mock_settings)
    orchestrator.cycles = 2

    assert orchestrator.settings.max_cycles > 0
    assert orchestrator.cycles >= orchestrator.settings.max_cycles


@pytest.mark.asyncio
async def test_orchestrator_cycle_increments(mock_settings: Settings) -> None:
    client = MockLLMClient(
        {
            "planner": '{"tasks": [{"id": "1", "description": "Task 1", "files": [], "dependencies": []}], "rationale": "Test"}',
            "worker": '{"files": [], "summary": "Done"}',
            "tester": '{"passed": true, "summary": "All tests passed", "failures": []}',
        },
        mock_settings,
    )
    orchestrator = Orchestrator(goal="Test goal", client=client, settings=mock_settings)

    with patch("furrow.agents.tester.TesterAgent.run", new_callable=AsyncMock) as mock_test:
        mock_test.return_value = TestResult(passed=True, summary="All tests passed", failures=[])
        await orchestrator._cycle()

    assert orchestrator.cycles == 0
    orchestrator.cycles += 1
    assert orchestrator.cycles == 1


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_writes_files(self, mock_settings: Settings) -> None:
        client = MockLLMClient(
            {
                "worker": '{"files": [{"path": "test_file.py", "content": "print(\'hello\')"}], "summary": "Created test file"}'
            },
            mock_settings,
        )

        task = TaskModel(id="1", description="Create a test file", files=["test_file.py"])
        worker = WorkerAgent(task=task, client=client, settings=mock_settings)

        with patch.object(client, "write_file", new_callable=AsyncMock) as mock_write:
            result = await worker.run()

        mock_write.assert_called_once()
        call_args = mock_write.call_args
        assert call_args[0][0] == "test_file.py"
        assert call_args[0][1] == "print('hello')"
        assert result == "Created test file"

    @pytest.mark.asyncio
    async def test_worker_handles_invalid_json(self, mock_settings: Settings) -> None:
        worker_response = {"default": "This is not valid JSON"}
        client = MockLLMClient(worker_response, mock_settings)

        task = TaskModel(id="1", description="Create a test file")
        worker = WorkerAgent(task=task, client=client, settings=mock_settings)

        result = await worker.run()

        assert "non-JSON response" in result

    @pytest.mark.asyncio
    async def test_worker_empty_files(self, mock_settings: Settings) -> None:
        worker_response = {"default": '{"files": [], "summary": "No changes needed"}'}
        client = MockLLMClient(worker_response, mock_settings)

        task = TaskModel(id="1", description="Do nothing")
        worker = WorkerAgent(task=task, client=client, settings=mock_settings)

        with patch.object(client, "write_file", new_callable=AsyncMock) as mock_write:
            result = await worker.run()

        mock_write.assert_not_called()
        assert result == "No changes needed"

    @pytest.mark.asyncio
    async def test_worker_multiple_files(self, mock_settings: Settings) -> None:
        worker_response = {
            "default": '{"files": [{"path": "file1.py", "content": "a=1"}, {"path": "file2.py", "content": "b=2"}], "summary": "Created two files"}'
        }
        client = MockLLMClient(worker_response, mock_settings)

        task = TaskModel(id="1", description="Create two files", files=["file1.py", "file2.py"])
        worker = WorkerAgent(task=task, client=client, settings=mock_settings)

        with patch.object(client, "write_file", new_callable=AsyncMock) as mock_write:
            result = await worker.run()

        assert mock_write.call_count == 2
        assert result == "Created two files"


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_planner_parses_valid_json(self, mock_settings: Settings) -> None:
        planner_response = {
            "default": '{"tasks": [{"id": "1", "description": "Task 1", "files": ["file1.py"], "dependencies": []}], "rationale": "Test rationale"}'
        }
        client = MockLLMClient(planner_response, mock_settings)

        planner = PlannerAgent(client=client, settings=mock_settings)
        plan = await planner.plan("Test goal")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"
        assert plan.tasks[0].description == "Task 1"
        assert plan.tasks[0].files == ["file1.py"]
        assert plan.rationale == "Test rationale"

    @pytest.mark.asyncio
    async def test_planner_raises_on_invalid_json(self, mock_settings: Settings) -> None:
        planner_response = {"default": "Not valid JSON"}
        client = MockLLMClient(planner_response, mock_settings)

        planner = PlannerAgent(client=client, settings=mock_settings)

        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await planner.plan("Test goal")

    @pytest.mark.asyncio
    async def test_planner_handles_multiple_tasks(self, mock_settings: Settings) -> None:
        planner_response = {
            "default": '{"tasks": [{"id": "1", "description": "Task 1", "files": ["a.py"], "dependencies": []}, {"id": "2", "description": "Task 2", "files": ["b.py"], "dependencies": ["1"]}, {"id": "3", "description": "Task 3", "files": ["c.py"], "dependencies": ["1", "2"]}], "rationale": "Three tasks"}'
        }
        client = MockLLMClient(planner_response, mock_settings)

        planner = PlannerAgent(client=client, settings=mock_settings)
        plan = await planner.plan("Test goal")

        assert len(plan.tasks) == 3
        assert plan.tasks[0].dependencies == []
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.tasks[2].dependencies == ["1", "2"]


class TestLLMClientRetry:
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, mock_settings: Settings) -> None:
        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Success on retry")]
            return mock_response

        settings = Settings()
        client = LLMClient(settings=settings)
        client._anthropic = AsyncMock()
        client._anthropic.messages.create = failing_then_success

        result = await client._complete_anthropic("test", "", "test-model")

        assert call_count == 3
        assert result == "Success on retry"

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, mock_settings: Settings) -> None:
        call_count = 0

        async def always_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent error")

        settings = Settings()
        client = LLMClient(settings=settings)
        client._anthropic = AsyncMock()
        client._anthropic.messages.create = always_fails

        with pytest.raises(ConnectionError):
            await client._complete_anthropic("test", "", "test-model")

        assert call_count == 3


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_parses_passed(self, mock_settings: Settings) -> None:
        tester_response = {"default": '{"passed": true, "summary": "All good", "failures": []}'}
        client = MockLLMClient(tester_response, mock_settings)

        from furrow.agents.tester import TesterAgent

        tester = TesterAgent(client=client, settings=mock_settings)

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "tests passed"
            result = await tester.run("Test goal", [])

        assert result.passed is True
        assert result.summary == "All good"

    @pytest.mark.asyncio
    async def test_tester_parses_failed(self, mock_settings: Settings) -> None:
        tester_response = {
            "default": '{"passed": false, "summary": "Failed", "failures": ["test_foo failed"]}'
        }
        client = MockLLMClient(tester_response, mock_settings)

        from furrow.agents.tester import TesterAgent

        tester = TesterAgent(client=client, settings=mock_settings)

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "test_foo failed"
            result = await tester.run("Test goal", [])

        assert result.passed is False
        assert "test_foo failed" in result.failures

    @pytest.mark.asyncio
    async def test_tester_handles_invalid_json(self, mock_settings: Settings) -> None:
        tester_response = {"default": "Not valid JSON but contains passed"}
        client = MockLLMClient(tester_response, mock_settings)

        from furrow.agents.tester import TesterAgent

        tester = TesterAgent(client=client, settings=mock_settings)

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "tests passed"
            result = await tester.run("Test goal", [])

        assert result.passed is True
