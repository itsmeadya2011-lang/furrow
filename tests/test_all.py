from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.cli.main import main


# ---------------------------------------------------------------------------
# config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_provider_enum(self):
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"
        assert Provider.OLLAMA == "ollama"

    def test_settings_defaults(self):
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.model == "claude-sonnet-4-20250514"
        assert s.max_parallel_tasks == 5
        assert s.max_cycles == 0
        assert s.workspace == Path.cwd()

    def test_settings_env_override(self, monkeypatch):
        monkeypatch.setenv("FURROW_MODEL", "gpt-4o")
        s = Settings()
        assert s.model == "gpt-4o"

    def test_task_model_defaults(self):
        t = TaskModel(id="1", description="do thing")
        assert t.files == []
        assert t.dependencies == []
        assert t.status == "pending"
        assert t.result is None

    def test_plan_model(self):
        p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        assert len(p.tasks) == 1
        assert p.rationale == "ok"

    def test_test_result_model(self):
        tr = TestResult(passed=True, summary="ok", failures=[])
        assert tr.passed is True
        assert tr.failures == []


# ---------------------------------------------------------------------------
# llm tests
# ---------------------------------------------------------------------------


class TestLLMClient:
    def test_init_defaults(self):
        c = LLMClient()
        assert c.settings == Settings()

    def test_init_with_settings(self):
        s = Settings(model="gpt-4o")
        c = LLMClient(settings=s)
        assert c.settings.model == "gpt-4o"

    def test_anthropic_provider_selection(self):
        c = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
        assert c.settings.provider == Provider.ANTHROPIC

    def test_openai_provider_selection(self):
        c = LLMClient(settings=Settings(provider=Provider.OPENAI))
        assert c.settings.provider == Provider.OPENAI

    def test_ollama_provider_selection(self):
        c = LLMClient(settings=Settings(provider=Provider.OLLAMA))
        assert c.settings.provider == Provider.OLLAMA

    def test_unsupported_provider_raises(self):
        c = LLMClient(settings=Settings(provider=Provider.OLLAMA))

        class FakeProvider:
            value = "fake"

        c.settings.provider = FakeProvider()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(c.complete("hello"))

    @pytest.mark.asyncio
    async def test_complete_anthropic(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        c = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
        c._anthropic = mock_client  # type: ignore[assignment]
        result = await c.complete("test prompt", model="claude-3-5-sonnet-20241022")
        assert result == "hello"
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_openai(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "hello"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        c = LLMClient(settings=Settings(provider=Provider.OPENAI))
        c._openai = mock_client  # type: ignore[assignment]
        result = await c.complete("test prompt", model="gpt-4o")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_complete_ollama(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "hello from ollama"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        c = LLMClient(settings=Settings(provider=Provider.OLLAMA))
        c._ollama = mock_client  # type: ignore[assignment]
        result = await c.complete("test prompt", model="llama3")
        assert result == "hello from ollama"

    @pytest.mark.asyncio
    async def test_read_write_file(self, tmp_path: Path):
        c = LLMClient()
        target = tmp_path / "test.txt"
        await c.write_file(target, "hello world")
        assert target.read_text() == "hello world"
        content = await c.read_file(target)
        assert content == "hello world"

    def test_write_file_creates_parents(self, tmp_path: Path):
        c = LLMClient()
        target = tmp_path / "a" / "b" / "test.txt"
        asyncio.run(c.write_file(target, "nested"))
        assert target.exists()
        assert target.read_text() == "nested"

    def test_list_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("c")
        c = LLMClient()
        files = c.list_files(tmp_path)
        assert "a.py" in files
        assert "b.txt" in files
        assert "sub/c.py" in files

    def test_list_files_empty_dir(self, tmp_path: Path):
        c = LLMClient()
        assert c.list_files(tmp_path) == []

    def test_list_files_missing_dir(self, tmp_path: Path):
        c = LLMClient()
        assert c.list_files(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# agents/planner tests
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_success(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
            "rationale": "ok"
        }))

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("build a thing")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do thing"
        mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_json_error(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="not json")

        agent = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("build a thing")


# ---------------------------------------------------------------------------
# agents/worker tests
# ---------------------------------------------------------------------------


class TestWorkerAgent:
    def test_parse_file_blocks_single(self):
        agent = WorkerAgent(task=TaskModel(id="1", description="x"))
        text = "FILE: src/main.py\nprint('hello')\n"
        blocks = agent._parse_file_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "src/main.py"
        assert "print('hello')" in blocks[0][1]

    def test_parse_file_blocks_multiple(self):
        agent = WorkerAgent(task=TaskModel(id="1", description="x"))
        text = "FILE: a.py\ncontent a\nFILE: b.py\ncontent b\n"
        blocks = agent._parse_file_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][0] == "a.py"
        assert blocks[1][0] == "b.py"

    def test_parse_file_blocks_none(self):
        agent = WorkerAgent(task=TaskModel(id="1", description="x"))
        assert agent._parse_file_blocks("no files here") == []

    @pytest.mark.asyncio
    async def test_run_writes_files(self, tmp_path: Path):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="FILE: hello.py\nprint('hi')\n")
        mock_client.write_file = AsyncMock()

        agent = WorkerAgent(
            task=TaskModel(id="1", description="create hello", files=["hello.py"]),
            client=mock_client,
        )
        result = await agent.run()
        assert "hello.py" in result
        mock_client.write_file.assert_called_once()
        mock_client.write_file.assert_called_with("hello.py", "print('hi')")

    @pytest.mark.asyncio
    async def test_run_no_files(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="No files needed")
        mock_client.write_file = AsyncMock()

        agent = WorkerAgent(task=TaskModel(id="1", description="review code"), client=mock_client)
        result = await agent.run()
        mock_client.write_file.assert_not_called()
        assert "Worker completed" in result


# ---------------------------------------------------------------------------
# agents/tester tests
# ---------------------------------------------------------------------------


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_with_passing_tests(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": True,
            "summary": "All good",
            "failures": []
        }))

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"ok", b""))
            proc.returncode = 0
            mock_subprocess.return_value = proc

            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])
            assert result.passed is True
            assert result.summary == "All good"

    @pytest.mark.asyncio
    async def test_run_with_failing_tests(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": False,
            "summary": "Failed",
            "failures": ["test_a failed"]
        }))

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"FAILURES", b""))
            proc.returncode = 1
            mock_subprocess.return_value = proc

            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])
            assert result.passed is False
            assert "test_a failed" in result.failures

    @pytest.mark.asyncio
    async def test_run_no_runner_found(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "no runner", "failures": []}')

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            agent = TesterAgent(client=mock_client)
            result = await agent.run("test goal", [])
            assert "No test runner found" in mock_client.complete.call_args[0][0]

    @pytest.mark.asyncio
    async def test_run_timeout_kills_process(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            proc = AsyncMock()
            proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            proc.kill = MagicMock()
            mock_subprocess.return_value = proc

            agent = TesterAgent(client=mock_client)
            # Should continue to next candidate after timeout
            with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
                result = await agent.run("test goal", [])
            proc.kill.assert_called()


# ---------------------------------------------------------------------------
# core/orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_is_done_no_plan(self):
        orch = Orchestrator(goal="test")
        assert orch._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_all_completed(self):
        orch = Orchestrator(goal="test")
        orch._current_plan = Plan(
            tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
            rationale="ok",
        )
        orch._current_plan.tasks[0].status = "completed"
        orch._current_plan.tasks[1].status = "completed"
        assert orch._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_some_failed(self):
        orch = Orchestrator(goal="test")
        orch._current_plan = Plan(
            tasks=[TaskModel(id="1", description="a")],
            rationale="ok",
        )
        orch._current_plan.tasks[0].status = "failed"
        assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_mixed_not_done(self):
        orch = Orchestrator(goal="test")
        orch._current_plan = Plan(
            tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
            rationale="ok",
        )
        orch._current_plan.tasks[0].status = "completed"
        orch._current_plan.tasks[1].status = "pending"
        assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_get_tasks_returns_plan_tasks(self):
        orch = Orchestrator(goal="test")
        plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
        orch._current_plan = plan
        assert orch._get_tasks() == plan.tasks

    @pytest.mark.asyncio
    async def test_cycle_runs_and_updates_goal(self):
        mock_plan = Plan(
            tasks=[TaskModel(id="1", description="do it", files=[])],
            rationale="ok",
        )
        mock_client = MagicMock()
        mock_planner = AsyncMock()
        mock_planner.plan = AsyncMock(return_value=mock_plan)
        mock_worker = AsyncMock(return_value="done")
        mock_tester = AsyncMock()
        mock_tester.run = AsyncMock(return_value=TestResult(passed=False, summary="fail", failures=["x"]))

        with patch.object(Orchestrator, "__init__", lambda self, goal, **kwargs: None):
            orch = Orchestrator.__new__(Orchestrator)
            orch.goal = "test"
            orch.client = mock_client
            orch.planner = mock_planner
            orch.cycles = 0
            orch._current_plan = None
            orch._console = MagicMock()

            with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
                with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                    await orch._cycle()

            assert orch.goal == "Fix failing tests:\nx"
            assert mock_worker.call_count == 1
            mock_tester.run.assert_called_once()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_start_with_goal(self):
        runner = CliRunner()
        with patch("furrow.cli.main.Orchestrator") as MockOrch:
            mock_orch = AsyncMock()
            MockOrch.return_value = mock_orch
            result = runner.invoke(main, ["start", "build a thing"])
            assert result.exit_code == 0
            mock_orch.run.assert_called_once()

    def test_start_with_model(self):
        runner = CliRunner()
        with patch("furrow.cli.main.Orchestrator") as MockOrch:
            mock_orch = AsyncMock()
            MockOrch.return_value = mock_orch
            result = runner.invoke(main, ["start", "build a thing", "--model", "gpt-4o"])
            assert result.exit_code == 0
            _, kwargs = MockOrch.call_args
            assert kwargs["settings"].model == "gpt-4o"

    def test_web_command(self):
        runner = CliRunner()
        with patch("furrow.cli.main.run") as mock_run:
            result = runner.invoke(main, ["web"])
            assert result.exit_code == 0
            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# web/server tests
# ---------------------------------------------------------------------------


class TestWebServer:
    def test_index_returns_html(self):
        from furrow.web.server import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "Furrow" in response.text

    @pytest.mark.asyncio
    async def test_websocket_streams_output(self):
        from furrow.web.server import app

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock()

        with patch("furrow.web.server.Orchestrator", return_value=mock_orch) as mock_orch_cls:
            client = TestClient(app)
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"goal": "test goal", "model": None})
                # The mocked orchestrator.run does not send anything, so we just
                # verify the endpoint accepted the connection and the orchestrator
                # was instantiated with the correct goal.
                mock_orch_cls.assert_called_once()
                call_kwargs = mock_orch_cls.call_args.kwargs
                assert call_kwargs["goal"] == "test goal"
