import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestLLMClient:
    @pytest.fixture
    def client(self):
        return LLMClient(settings=Settings(provider="openai", openai_api_key="test"))

    def test_list_files_empty(self, client):
        assert client.list_files("/nonexistent/path") == []

    def test_write_and_read_file(self, client, tmp_path):
        target = tmp_path / "subdir" / "test.txt"
        asyncio.run(client.write_file(target, "hello world"))
        result = asyncio.run(client.read_file(target))
        assert result == "hello world"


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_success(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = '{"tasks": [{"id": "1", "description": "do x", "files": [], "dependencies": []}], "rationale": "ok"}'
        planner = PlannerAgent(client=mock_client)
        plan = await planner.plan("build a house")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do x"

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = "not json"
        planner = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("build a house")


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_prompts_correctly(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = "Implemented task"
        task = TaskModel(id="1", description="add auth", files=["src/auth.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "Implemented task"
        prompt = mock_client.complete.call_args[0][0]
        assert "add auth" in prompt
        assert "src/auth.py" in prompt


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_with_failing_subprocess(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = '{"passed": false, "summary": "fail", "failures": ["x failed"]}'
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            proc = AsyncMock()
            proc.communicate.return_value = (b"fail", b"")
            proc.returncode = 1
            mock_subprocess.return_value = proc
            tester = TesterAgent(client=mock_client)
            result = await tester.run("do tests", [])
        assert result.passed is False
        assert result.failures == ["x failed"]

    @pytest.mark.asyncio
    async def test_run_no_test_runner(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = '{"passed": true, "summary": "no tests", "failures": []}'
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            tester = TesterAgent(client=mock_client)
            result = await tester.run("do tests", [])
        assert "No test runner found" in (mock_client.complete.call_args[0][0])


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_is_done_with_no_plan(self):
        orchestrator = Orchestrator(goal="test")
        assert not orchestrator._is_done()

    @pytest.mark.asyncio
    async def test_is_done_with_completed_tasks(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="x", status="completed")],
            rationale="ok",
        )
        assert orchestrator._is_done()

    @pytest.mark.asyncio
    async def test_is_done_with_failed_tasks(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="x", status="failed")],
            rationale="ok",
        )
        assert not orchestrator._is_done()

    @pytest.mark.asyncio
    async def test_is_done_with_pending_tasks(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator.plan = Plan(
            tasks=[TaskModel(id="1", description="x", status="pending")],
            rationale="ok",
        )
        assert not orchestrator._is_done()

    @pytest.mark.asyncio
    async def test_max_cycles_halts(self):
        orchestrator = Orchestrator(goal="test", settings=Settings(max_cycles=1))
        planner_mock = AsyncMock()
        planner_mock.plan.return_value = Plan(tasks=[], rationale="none")
        orchestrator.planner = planner_mock
        await orchestrator.run()
        assert orchestrator.cycles == 1
