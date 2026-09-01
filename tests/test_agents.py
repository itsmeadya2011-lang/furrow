import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        max_parallel_tasks=5,
        max_cycles=0,
    )


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_success(self, mock_settings: Settings):
        client = LLMClient(mock_settings)
        client.complete = AsyncMock(
            return_value='{"tasks": [{"id": "1", "description": "do x", "files": [], "dependencies": []}], "rationale": "ok"}'
        )
        agent = PlannerAgent(client=client)
        plan = await agent.plan("build a thing")
        assert plan.rationale == "ok"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises(self, mock_settings: Settings):
        client = LLMClient(mock_settings)
        client.complete = AsyncMock(return_value="not json at all")
        agent = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("build a thing")


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_writes_files(self, mock_settings: Settings):
        client = LLMClient(mock_settings)
        client.complete = AsyncMock(
            return_value='{"summary": "done", "files": {"test_output.py": "print(1)"}}'
        )
        client.write_file = AsyncMock()
        task = TaskModel(id="1", description="write a file", files=[])
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()
        assert "done" in result
        client.write_file.assert_awaited_once_with("test_output.py", "print(1)")

    @pytest.mark.asyncio
    async def test_worker_read_files_for_context(self, mock_settings: Settings):
        client = LLMClient(mock_settings)
        client.complete = AsyncMock(return_value='{"summary": "ok", "files": {}}')
        client.read_file = AsyncMock(return_value="existing content")
        client.write_file = AsyncMock()
        task = TaskModel(id="1", description="edit file", files=["existing.py"])
        agent = WorkerAgent(task=task, client=client)
        await agent.run()
        client.read_file.assert_awaited_once_with("existing.py")

    @pytest.mark.asyncio
    async def test_worker_fallback_to_raw_text(self, mock_settings: Settings):
        client = LLMClient(mock_settings)
        client.complete = AsyncMock(return_value="raw text that is not JSON")
        client.write_file = AsyncMock()
        task = TaskModel(id="1", description="task here", files=[])
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()
        assert result == "raw text that is not JSON"
        client.write_file.assert_not_called()


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_parses_json(self, mock_settings: Settings, monkeypatch):
        client = LLMClient(mock_settings)
        client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "all good", "failures": []}'
        )

        async def fake_run_tests():
            return "test output here"

        monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="x", files=[])
        result = await agent.run("goal", [task])
        assert result.passed is True
        assert result.summary == "all good"

    @pytest.mark.asyncio
    async def test_tester_fallback_raw_text(self, mock_settings: Settings, monkeypatch):
        client = LLMClient(mock_settings)
        client.complete = AsyncMock(return_value="tests passed, yay")

        async def fake_run_tests():
            return "output"

        monkeypatch.setattr(TesterAgent, "_run_tests", fake_run_tests)
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="x", files=[])
        result = await agent.run("goal", [task])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_tester_handles_test_runner_error(self, mock_settings: Settings, monkeypatch):
        client = LLMClient(mock_settings)

        async def boom():
            raise FileNotFoundError("no runner")

        monkeypatch.setattr(TesterAgent, "_run_tests", boom)
        agent = TesterAgent(client=client)
        task = TaskModel(id="1", description="x", files=[])
        result = await agent.run("goal", [task])
        assert result.passed is False
        assert "no runner" in result.failures[0]
