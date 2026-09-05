from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.log_level == "INFO"


class TestOrchestrator:
    def test_get_tasks_returns_current_plan_tasks(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator._current_plan = None
        assert orchestrator._get_tasks() == []

        plan = Plan(
            tasks=[TaskModel(id="1", description="do thing")],
            rationale="ok",
        )
        orchestrator._current_plan = plan
        assert len(orchestrator._get_tasks()) == 1
        assert orchestrator._get_tasks()[0].description == "do thing"

    def test_is_done_with_empty_tasks(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator._current_plan = Plan(tasks=[], rationale="ok")
        assert orchestrator._is_done() is True

    def test_is_done_with_all_completed(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is True

    def test_is_done_with_some_pending(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    def test_is_done_with_failed_task(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="ok",
        )
        assert orchestrator._is_done() is False

    def test_is_done_no_plan(self):
        orchestrator = Orchestrator(goal="test")
        orchestrator._current_plan = None
        assert orchestrator._is_done() is False


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_routes_to_anthropic(self):
        settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=settings)
        client._anthropic = AsyncMock()
        client._anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text="anthropic response")]
        )
        result = await client.complete("hello")
        assert result == "anthropic response"
        client._anthropic.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_routes_to_openai(self):
        settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=settings)
        client._openai = AsyncMock()
        client._openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="openai response"))]
        )
        result = await client.complete("hello")
        assert result == "openai response"
        client._openai.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_routes_to_ollama(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:11434")
        client = LLMClient(settings=settings)
        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ollama response"))]
        )
        with patch("furrow.llm.AsyncOpenAI", return_value=mock_openai):
            result = await client.complete("hello")
        assert result == "ollama response"
        mock_openai.chat.completions.create.assert_called_once()


class TestOllamaSupport:
    @pytest.mark.asyncio
    async def test_complete_ollama_initializes_with_base_url(self):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:11434")
        client = LLMClient(settings=settings)
        assert client._openai is None

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        with patch("furrow.llm.AsyncOpenAI", return_value=mock_openai) as mock_cls:
            result = await client.complete("hello")

        assert result == "ok"
        mock_cls.assert_called_once_with(api_key="ollama", base_url="http://custom:11434")
        mock_openai.chat.completions.create.assert_called_once()


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_includes_workspace_files(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.list_files.return_value = ["src/main.py", "README.md"]
        mock_client.complete.return_value = json.dumps(
            {
                "tasks": [{"id": "1", "description": "do thing", "files": ["src/main.py"], "dependencies": []}],
                "rationale": "ok",
            }
        )

        planner = PlannerAgent(client=mock_client, workspace=Path("/fake/workspace"))
        plan = await planner.plan("build a thing")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"
        called_with = mock_client.complete.call_args
        prompt = called_with[0][0]
        assert "README.md" in prompt
        assert "src/main.py" in prompt

    @pytest.mark.asyncio
    async def test_plan_handles_empty_workspace(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.list_files.return_value = []
        mock_client.complete.return_value = json.dumps(
            {
                "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
                "rationale": "ok",
            }
        )

        planner = PlannerAgent(client=mock_client, workspace=Path("/fake/workspace"))
        plan = await planner.plan("build a thing")

        assert len(plan.tasks) == 1
        called_with = mock_client.complete.call_args
        prompt = called_with[0][0]
        assert "(empty directory)" in prompt

    @pytest.mark.asyncio
    async def test_plan_handles_list_files_exception(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.list_files.side_effect = RuntimeError("boom")
        mock_client.complete.return_value = json.dumps(
            {
                "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
                "rationale": "ok",
            }
        )

        planner = PlannerAgent(client=mock_client, workspace=Path("/fake/workspace"))
        plan = await planner.plan("build a thing")

        assert len(plan.tasks) == 1
        called_with = mock_client.complete.call_args
        prompt = called_with[0][0]
        assert "(could not list files)" in prompt


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_includes_file_contents(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.read_file.return_value = "print('hello')"
        mock_client.complete.return_value = "worker summary"

        task = TaskModel(id="1", description="fix bug", files=["main.py"])
        worker = WorkerAgent(task=task, client=mock_client, workspace=Path("/fake/workspace"))
        result = await worker.run()

        assert result == "worker summary"
        mock_client.read_file.assert_called_once_with(Path("/fake/workspace/main.py"))
        called_with = mock_client.complete.call_args
        prompt = called_with[0][0]
        assert "print('hello')" in prompt
        assert "fix bug" in prompt

    @pytest.mark.asyncio
    async def test_run_handles_read_file_error(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.read_file.side_effect = RuntimeError("file not found")
        mock_client.complete.return_value = "worker summary"

        task = TaskModel(id="1", description="fix bug", files=["main.py"])
        worker = WorkerAgent(task=task, client=mock_client, workspace=Path("/fake/workspace"))
        result = await worker.run()

        assert result == "worker summary"
        called_with = mock_client.complete.call_args
        prompt = called_with[0][0]
        assert "(could not read file)" in prompt

    @pytest.mark.asyncio
    async def test_run_no_workspace_skips_files(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = "worker summary"

        task = TaskModel(id="1", description="fix bug", files=["main.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "worker summary"
        mock_client.read_file.assert_not_called()


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_uses_cwd_workspace(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        settings = Settings(workspace=tmp_path)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings
        mock_client.complete.return_value = json.dumps(
            {"passed": True, "summary": "ok", "failures": []}
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        async def fake_communicate():
            return b"", b""

        mock_proc.communicate = fake_communicate

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            assert kwargs.get("cwd") == tmp_path
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            agent = TesterAgent(client=mock_client)
            result = await agent.run("goal", [])

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_tests_detects_pyproject(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        settings = Settings(workspace=tmp_path)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings
        mock_client.complete.return_value = json.dumps(
            {"passed": True, "summary": "ok", "failures": []}
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        async def fake_communicate():
            return b"", b""

        mock_proc.communicate = fake_communicate

        called_commands = []

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            called_commands.append(cmd)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            agent = TesterAgent(client=mock_client)
            await agent.run("goal", [])

        assert any(cmd == ("python", "-m", "pytest", "-q") for cmd in called_commands)

    @pytest.mark.asyncio
    async def test_run_tests_timeout_kills_process_and_continues(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        settings = Settings(workspace=tmp_path)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings
        mock_client.complete.return_value = json.dumps(
            {"passed": True, "summary": "ok", "failures": []}
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.kill.return_value = None

        call_count = 0

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                async def timeout_communicate():
                    raise asyncio.TimeoutError()
                mock_proc.communicate = timeout_communicate
            else:
                async def ok_communicate():
                    return b"", b""
                mock_proc.communicate = ok_communicate
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            agent = TesterAgent(client=mock_client)
            result = await agent._run_tests()

        assert "[exit code: 0]" in result
        assert call_count >= 2
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_tests_handles_file_not_found(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("not found")):
            agent = TesterAgent(client=mock_client)
            result = await agent._run_tests()

        assert "No test runner found." in result

    @pytest.mark.asyncio
    async def test_run_tests_detects_package_json(self, tmp_path):
        package = tmp_path / "package.json"
        package.write_text('{"name": "test"}')

        settings = Settings(workspace=tmp_path)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings
        mock_client.complete.return_value = json.dumps(
            {"passed": True, "summary": "ok", "failures": []}
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        async def fake_communicate():
            return b"", b""

        mock_proc.communicate = fake_communicate

        called_commands = []

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            called_commands.append(cmd)
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            agent = TesterAgent(client=mock_client)
            await agent.run("goal", [])

        assert any(cmd == ("npm", "test", "--", "--silent") for cmd in called_commands)

    @pytest.mark.asyncio
    async def test_run_tests_all_timeout_returns_fallback(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        settings = Settings(workspace=tmp_path)
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings

        mock_proc = AsyncMock()
        mock_proc.kill.return_value = None

        async def fake_communicate():
            raise asyncio.TimeoutError()

        mock_proc.communicate = fake_communicate

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            agent = TesterAgent(client=mock_client)
            result = await agent._run_tests()

        assert "No test runner found. Consider installing pytest." in result
