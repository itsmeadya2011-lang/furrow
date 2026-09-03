from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestConfig:
    def test_plan_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"

    def test_test_result(self):
        t = TestResult(passed=True, summary="ok", failures=[])
        assert t.passed is True


class TestLLMClient:
    @pytest.fixture
    def client(self):
        return LLMClient(settings=Settings(provider=Provider.ANTHROPIC, model="test-model"))

    @pytest.mark.asyncio
    async def test_complete_unsupported_provider(self, client):
        client.settings.provider = Provider.OLLAMA
        client.settings.ollama_base_url = "http://localhost:11434"
        with patch("httpx.AsyncClient.post", side_effect=Exception("connection refused")):
            with pytest.raises(RuntimeError, match="Unexpected error calling Ollama"):
                await client.complete("hello", model="llama3")

    @pytest.mark.asyncio
    async def test_complete_ollama_success(self):
        settings = Settings(provider=Provider.OLLAMA, model="llama3", ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "hello from ollama"}}
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await client.complete("hi", model="llama3")
        assert result == "hello from ollama"

    @pytest.mark.asyncio
    async def test_complete_ollama_http_error(self):
        settings = Settings(provider=Provider.OLLAMA, model="llama3", ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        with patch("httpx.AsyncClient.post", side_effect=Exception("500 Server Error")):
            with pytest.raises(RuntimeError, match="Unexpected error calling Ollama"):
                await client.complete("hi", model="llama3")

    @pytest.mark.asyncio
    async def test_read_write_file(self, tmp_path: Path):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model")
        client = LLMClient(settings=settings)
        target = tmp_path / "sub" / "file.txt"
        await client.write_file(target, "hello world")
        assert target.read_text() == "hello world"
        content = await client.read_file(target)
        assert content == "hello world"

    def test_list_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("c")
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model")
        client = LLMClient(settings=settings)
        files = sorted(client.list_files(tmp_path))
        assert files == ["a.py", "b.txt", "sub/c.py"]


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_parses_json(self):
        client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, model="test-model"))
        agent = PlannerAgent(client=client)
        response = json.dumps({
            "tasks": [{"id": "1", "description": "add auth", "files": ["auth.py"], "dependencies": []}],
            "rationale": "simple",
        })
        with patch.object(client, "complete", new_callable=AsyncMock, return_value=response):
            plan = await agent.plan("add auth")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_returns_summary(self):
        client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, model="test-model"))
        task = TaskModel(id="1", description="do work", files=["foo.py"])
        agent = WorkerAgent(task=task, client=client)
        with patch.object(client, "complete", new_callable=AsyncMock, return_value="done"):
            result = await agent.run()
        assert result == "done"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_no_runner(self):
        client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, model="test-model"))
        agent = TesterAgent(client=client)
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            output = await agent._run_tests()
        assert output == "No test runner found."

    @pytest.mark.asyncio
    async def test_run_tests_timeout(self):
        client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, model="test-model"))
        agent = TesterAgent(client=client)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                output = await agent._run_tests()
        assert output == "No test runner found."

    @pytest.mark.asyncio
    async def test_run_returns_failure_on_test_error(self):
        client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, model="test-model"))
        agent = TesterAgent(client=client)
        with patch.object(agent, "_run_tests", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            result = await agent.run("goal", [])
        assert result.passed is False
        assert "boom" in result.summary


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_max_cycles_enforced(self):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model", max_cycles=2)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="do stuff", client=client)
        plan = Plan(tasks=[], rationale="done")

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=TestResult(passed=True, summary="ok", failures=[])):
                await orchestrator.run()

        assert orchestrator.cycles == 1

    @pytest.mark.asyncio
    async def test_get_tasks_returns_plan_tasks(self):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model", max_cycles=1)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="do stuff", client=client)
        tasks = [TaskModel(id="1", description="task1"), TaskModel(id="2", description="task2")]
        plan = Plan(tasks=tasks, rationale="ok")

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=TestResult(passed=True, summary="ok", failures=[])):
                await orchestrator.run()

        assert orchestrator._get_tasks() == tasks

    @pytest.mark.asyncio
    async def test_is_done_when_no_tasks(self):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model", max_cycles=1)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="do stuff", client=client)
        plan = Plan(tasks=[], rationale="done")

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=TestResult(passed=True, summary="ok", failures=[])):
                await orchestrator.run()

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_when_all_completed(self):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model", max_cycles=1)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="do stuff", client=client)
        tasks = [TaskModel(id="1", description="task1"), TaskModel(id="2", description="task2")]
        plan = Plan(tasks=tasks, rationale="ok")
        for t in tasks:
            t.status = "completed"

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=TestResult(passed=True, summary="ok", failures=[])):
                await orchestrator.run()

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_when_any_failed(self):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model", max_cycles=1)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="do stuff", client=client)
        tasks = [TaskModel(id="1", description="task1"), TaskModel(id="2", description="task2")]
        tasks[0].status = "completed"
        tasks[1].status = "failed"
        plan = Plan(tasks=tasks, rationale="ok")

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=TestResult(passed=True, summary="ok", failures=[])):
                await orchestrator.run()

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_max_cycles_halts_loop(self):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model", max_cycles=2)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="do stuff", client=client)
        tasks = [TaskModel(id="1", description="task1")]
        plan = Plan(tasks=tasks, rationale="ok")
        # Tasks never complete, so _is_done() returns False and loop continues until max_cycles
        for t in tasks:
            t.status = "pending"

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=TestResult(passed=False, summary="fail", failures=["fail"])):
                await orchestrator.run()

        assert orchestrator.cycles == 2

    @pytest.mark.asyncio
    async def test_planning_failure_stops_orchestrator(self):
        settings = Settings(provider=Provider.ANTHROPIC, model="test-model", max_cycles=5)
        client = LLMClient(settings=settings)
        orchestrator = Orchestrator(goal="do stuff", client=client)

        with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, side_effect=ValueError("bad plan")):
            await orchestrator.run()

        assert orchestrator.cycles == 1


class TestCLI:
    def test_config_command(self):
        runner = CliRunner()
        result = runner.invoke("furrow", ["config"])
        assert result.exit_code == 0
        assert "provider" in result.output
