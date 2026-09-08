from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

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

    def test_provider_enum(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_settings_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0


class TestLLMClient:
    def test_provider_routing_anthropic(self):
        client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            async def run():
                return await client.complete("hello")
            import asyncio
            asyncio.run(run())
            mock.assert_called_once()

    def test_provider_routing_openai(self):
        client = LLMClient(settings=Settings(provider=Provider.OPENAI))
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock:
            async def run():
                return await client.complete("hello")
            import asyncio
            asyncio.run(run())
            mock.assert_called_once()

    def test_provider_routing_ollama(self):
        client = LLMClient(settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock:
            async def run():
                return await client.complete("hello")
            import asyncio
            asyncio.run(run())
            mock.assert_called_once()

    def test_unsupported_provider(self):
        client = LLMClient(settings=Settings(provider="invalid"))  # type: ignore
        with pytest.raises(ValueError, match="Unsupported provider"):
            async def run():
                await client.complete("hello")
            import asyncio
            asyncio.run(run())

    def test_list_files_empty_dir(self, tmp_path: Path):
        client = LLMClient(settings=Settings(workspace=tmp_path))
        assert client.list_files(tmp_path) == []

    def test_list_files_with_files(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# test")
        client = LLMClient(settings=Settings(workspace=tmp_path))
        files = client.list_files(tmp_path)
        assert "src/main.py" in files
        assert "README.md" in files


class TestOrchestrator:
    def test_init_stores_original_goal(self):
        orch = Orchestrator(goal="test goal")
        assert orch.original_goal == "test goal"
        assert orch.goal == "test goal"
        assert orch.current_plan is None
        assert orch.cycles == 0

    def test_get_tasks_empty_before_plan(self):
        orch = Orchestrator(goal="test")
        assert orch._get_tasks() == []

    def test_is_done_no_tasks(self):
        orch = Orchestrator(goal="test")
        assert orch._is_done() is True

    def test_is_done_with_completed_tasks(self):
        orch = Orchestrator(goal="test")
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is True

    def test_is_done_with_failed_tasks(self):
        orch = Orchestrator(goal="test")
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is False

    def test_is_done_with_pending_tasks(self):
        orch = Orchestrator(goal="test")
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="pending"),
            ],
            rationale="ok",
        )
        assert orch._is_done() is False

    def test_max_cycles_logic(self):
        orch = Orchestrator(goal="test")
        orch.cycles = 3
        orch.settings = Settings(max_cycles=3)
        # Condition from run(): settings.max_cycles > 0 and self.cycles > settings.max_cycles
        assert not (orch.settings.max_cycles > 0 and orch.cycles > orch.settings.max_cycles)
        orch.cycles = 4
        assert orch.settings.max_cycles > 0 and orch.cycles > orch.settings.max_cycles


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_includes_file_tree(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hi')")
        client = LLMClient(settings=Settings(workspace=tmp_path))
        agent = PlannerAgent(client=client)
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = json.dumps({
                "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
                "rationale": "ok",
            })
            plan = await agent.plan("test goal")
            call_args = mock_complete.call_args[0][0]
            assert "Workspace file tree:" in call_args
            assert "main.py" in call_args
            assert "test goal" in call_args

    @pytest.mark.asyncio
    async def test_plan_includes_original_goal(self, tmp_path: Path):
        client = LLMClient(settings=Settings(workspace=tmp_path))
        agent = PlannerAgent(client=client)
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = json.dumps({
                "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
                "rationale": "ok",
            })
            await agent.plan("fix tests", original_goal="add auth")
            call_args = mock_complete.call_args[0][0]
            assert "Original goal: add auth" in call_args


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_reads_files_and_writes_changes(self, tmp_path: Path):
        (tmp_path / "hello.py").write_text("print('old')")
        task = TaskModel(id="1", description="update hello.py", files=["hello.py"], dependencies=[], status="pending")
        client = LLMClient(settings=Settings(workspace=tmp_path))
        agent = WorkerAgent(task=task, client=client)
        response = json.dumps({
            "files": {"hello.py": "print('new')"},
            "summary": "Updated hello.py",
        })
        with patch.object(client, "complete", new_callable=AsyncMock, return_value=response):
            result = await agent.run()
        assert result == "Updated hello.py"
        assert (tmp_path / "hello.py").read_text() == "print('new')"

    @pytest.mark.asyncio
    async def test_run_no_files_listed_reads_workspace(self, tmp_path: Path):
        (tmp_path / "foo.py").write_text("x=1")
        task = TaskModel(id="1", description="do something", files=[], dependencies=[], status="pending")
        client = LLMClient(settings=Settings(workspace=tmp_path))
        agent = WorkerAgent(task=task, client=client)
        response = json.dumps({"files": {}, "summary": "nothing to do"})
        with patch.object(client, "complete", new_callable=AsyncMock, return_value=response) as mock:
            result = await agent.run()
            call_args = mock.call_args[0][0]
            assert "foo.py" in call_args

    @pytest.mark.asyncio
    async def test_run_returns_raw_on_bad_json(self, tmp_path: Path):
        task = TaskModel(id="1", description="do thing", files=[], dependencies=[], status="pending")
        client = LLMClient(settings=Settings(workspace=tmp_path))
        agent = WorkerAgent(task=task, client=client)
        with patch.object(client, "complete", new_callable=AsyncMock, return_value="not json"):
            result = await agent.run()
        assert result == "not json"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_passes_cwd(self, tmp_path: Path):
        client = LLMClient(settings=Settings(workspace=tmp_path))
        agent = TesterAgent(client=client)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
            mock_subprocess.return_value = mock_proc
            result = await agent._run_tests()
            call_kwargs = mock_subprocess.call_args.kwargs
            assert call_kwargs["cwd"] == str(tmp_path)
