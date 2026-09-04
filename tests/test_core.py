import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.model == "claude-sonnet-4-20250514"
        assert s.planner_model == "claude-3-5-haiku-20241022"
        assert s.worker_model == "claude-3-5-sonnet-20241022"
        assert s.tester_model == "claude-3-5-sonnet-20241022"
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None
        assert s.ollama_base_url == "http://localhost:11434"
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert isinstance(s.workspace, Path)
        assert s.log_level == "INFO"

    def test_provider_enum(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"


class TestTaskModel:
    def test_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.id == "1"
        assert t.description == "do thing"
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_with_values(self):
        t = TaskModel(
            id="2",
            description="write tests",
            files=["tests/test_x.py"],
            dependencies=["1"],
            status="done",
            result="done",
        )
        assert t.files == ["tests/test_x.py"]
        assert t.dependencies == ["1"]
        assert t.status == "done"
        assert t.result == "done"


class TestPlan:
    def test_parse(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert p.tasks[0].description == "do thing"
        assert p.rationale == "ok"

    def test_invalid_plan_raises(self):
        with pytest.raises(ValidationError):
            Plan(rationale="missing tasks")


class TestTestResult:
    def test_defaults(self):
        t = TestResult(passed=True, summary="ok")
        assert t.passed is True
        assert t.summary == "ok"
        assert t.failures == []

    def test_with_failures(self):
        t = TestResult(passed=False, summary="boom", failures=["test_a failed"])
        assert t.passed is False
        assert t.failures == ["test_a failed"]


class TestLLMClient:
    def test_lazy_init(self):
        client = LLMClient()
        assert client._anthropic is None
        assert client._openai is None
        assert client._openai_ollama is None

    def test_anthropic_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = LLMClient()
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            _ = client.anthropic

    def test_openai_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient()
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            _ = client.openai

    def test_openai_ollama_no_key_succeeds(self):
        client = LLMClient()
        assert client.openai_ollama is not None

    @pytest.mark.asyncio
    async def test_complete_routes_anthropic(self):
        client = LLMClient()
        with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = "anthropic result"
            client.settings.provider = Provider.ANTHROPIC
            result = await client.complete("prompt")
            assert result == "anthropic result"

    @pytest.mark.asyncio
    async def test_complete_routes_openai(self):
        client = LLMClient()
        with patch.object(client, "_complete_openai", new_callable=AsyncMock) as mock:
            mock.return_value = "openai result"
            client.settings.provider = Provider.OPENAI
            result = await client.complete("prompt")
            assert result == "openai result"

    @pytest.mark.asyncio
    async def test_complete_routes_ollama(self):
        client = LLMClient()
        with patch.object(client, "_complete_ollama", new_callable=AsyncMock) as mock:
            mock.return_value = "ollama result"
            client.settings.provider = Provider.OLLAMA
            result = await client.complete("prompt")
            assert result == "ollama result"

    def test_list_files_empty(self, tmp_path):
        client = LLMClient()
        assert client.list_files(tmp_path) == []

    def test_list_files_populated(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("b")
        client = LLMClient()
        files = sorted(client.list_files(tmp_path))
        assert files == ["a.py", "sub/b.py"]

    def test_list_files_missing(self, tmp_path):
        client = LLMClient()
        assert client.list_files(tmp_path / "nonexistent") == []

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello")
        client = LLMClient()
        assert await client.read_file(f) == "hello"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tmp_path):
        client = LLMClient()
        target = tmp_path / "deep" / "nested" / "out.txt"
        await client.write_file(target, "content")
        assert target.read_text() == "content"


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_valid_json(self):
        response = json.dumps({
            "tasks": [{"id": "1", "description": "build api"}],
            "rationale": "simple",
        })
        client = AsyncMock()
        client.complete = AsyncMock(return_value=response)
        agent = PlannerAgent(client=client)
        plan = await agent.plan("build an api")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "1"
        assert plan.rationale == "simple"

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises(self):
        client = AsyncMock()
        client.complete = AsyncMock(return_value="not json")
        agent = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("build an api")

    @pytest.mark.asyncio
    async def test_plan_missing_fields_raises(self):
        response = json.dumps({"tasks": []})
        client = AsyncMock()
        client.complete = AsyncMock(return_value=response)
        agent = PlannerAgent(client=client)
        with pytest.raises((ValueError, ValidationError)):
            await agent.plan("build an api")


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_prompt_contains_task(self):
        task = TaskModel(id="1", description="add auth", files=["auth.py"])
        client = AsyncMock()
        client.complete = AsyncMock(return_value="done")
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()
        assert result == "done"
        call_args = client.complete.call_args
        prompt = call_args.args[0]
        assert "add auth" in prompt
        assert "auth.py" in prompt
        assert call_args.kwargs["model"] == client.settings.worker_model

    @pytest.mark.asyncio
    async def test_run_no_files(self):
        task = TaskModel(id="1", description="cleanup")
        client = AsyncMock()
        client.complete = AsyncMock(return_value="done")
        agent = WorkerAgent(task=task, client=client)
        await agent.run()
        prompt = client.complete.call_args.args[0]
        assert "cleanup" in prompt
        assert "any" in prompt


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_pass_path(self):
        client = AsyncMock()
        client.settings.tester_model = "test-model"
        agent = TesterAgent(client=client)
        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="pytest passed"):
            with patch.object(agent, "_evaluate", new_callable=AsyncMock) as eval_mock:
                eval_mock.return_value = TestResult(passed=True, summary="ok", failures=[])
                result = await agent.run("goal", [])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fail_path_no_fix(self):
        client = AsyncMock()
        client.settings.tester_model = "test-model"
        agent = TesterAgent(client=client)
        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="pytest failed"):
            with patch.object(agent, "_evaluate", new_callable=AsyncMock) as eval_mock:
                eval_mock.return_value = TestResult(passed=False, summary="1 failed", failures=["oops"])
                with patch.object(agent, "_attempt_fix", new_callable=AsyncMock) as fix_mock:
                    fix_mock.return_value = TestResult(passed=False, summary="still broken", failures=["oops"])
                    result = await agent.run("goal", [])
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_fix_worker_spawned(self):
        client = AsyncMock()
        client.settings.tester_model = "test-model"
        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task", files=["main.py"])]
        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="pytest failed"):
            with patch.object(agent, "_evaluate", new_callable=AsyncMock) as eval_mock:
                eval_mock.return_value = TestResult(passed=False, summary="1 failed", failures=["oops"])
                with patch.object(agent, "_attempt_fix", new_callable=AsyncMock) as fix_mock:
                    fix_mock.return_value = TestResult(passed=True, summary="fixed", failures=[])
                    result = await agent.run("goal", tasks)
        assert result.passed is True
        fix_mock.assert_called_once()
