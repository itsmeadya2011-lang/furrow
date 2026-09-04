import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from furrow.agents.tester import TesterAgent
from furrow.cli.main import main
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ─── Orchestrator ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_tasks_returns_tasks_set_after_cycle():
    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")

    orchestrator = Orchestrator(goal="do stuff", client=client, settings=settings)

    with patch.object(
        orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan
    ), patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orchestrator._cycle()

    tasks = orchestrator._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].description == "do thing"


def test_is_done_returns_false_when_no_tasks():
    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(goal="do stuff", client=client, settings=settings)
    assert orchestrator._is_done() is False


def test_is_done_returns_true_when_all_tasks_completed():
    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(goal="do stuff", client=client, settings=settings)
    orchestrator._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_is_done_returns_false_when_any_task_failed():
    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(goal="do stuff", client=client, settings=settings)
    orchestrator._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_returns_false_when_tasks_not_all_completed():
    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(goal="do stuff", client=client, settings=settings)
    orchestrator._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_stops_after_max_cycles():
    settings = Settings(max_cycles=2)
    client = LLMClient(settings=settings)
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")

    orchestrator = Orchestrator(goal="do stuff", client=client, settings=settings)

    with patch.object(
        orchestrator.planner, "plan", new_callable=AsyncMock, return_value=plan
    ), patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester, patch.object(
        orchestrator, "_is_done", return_value=False
    ):
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orchestrator.run()

    assert orchestrator.cycles == 2


# ─── LLMClient ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_write_file_utf8(tmp_path):
    client = LLMClient()
    target = tmp_path / "utf8.txt"
    content = "Hello, 世界 🌍"

    await client.write_file(target, content)
    read_back = await client.read_file(target)

    assert read_back == content


def test_list_files_returns_empty_for_nonexistent_directory():
    client = LLMClient()
    result = client.list_files("/nonexistent/path/that/does/not/exist")
    assert result == []


def test_complete_raises_value_error_for_unsupported_provider():
    settings = Settings()
    client = LLMClient(settings=settings)
    with patch.object(settings, "provider", "unsupported"):
        with pytest.raises(ValueError, match="Unsupported provider"):
            asyncio.run(client.complete("test prompt"))


# ─── TesterAgent ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_tests_returns_no_runner_when_not_found():
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        agent = TesterAgent()
        result = await agent._run_tests()
    assert result == "No test runner found."


@pytest.mark.asyncio
async def test_run_tests_handles_timeout():
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()
        mock_exec.return_value = mock_proc

        agent = TesterAgent()
        result = await agent._run_tests()

    assert result == "No test runner found."
    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_run_returns_failing_result_when_run_tests_raises():
    agent = TesterAgent()
    with patch.object(
        agent, "_run_tests", new_callable=AsyncMock, side_effect=RuntimeError("boom")
    ):
        result = await agent.run("do stuff", [])

    assert result.passed is False
    assert "boom" in result.summary


# ─── CLI ─────────────────────────────────────────────────────────────────────


def test_main_has_version_option():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "furrow" in result.output.lower()


@pytest.mark.asyncio
async def test_start_command_calls_orchestrator_with_goal():
    runner = CliRunner()
    with patch("furrow.cli.main.Orchestrator") as MockOrchestrator:
        mock_orch_instance = AsyncMock()
        MockOrchestrator.return_value = mock_orch_instance
        result = runner.invoke(main, ["start", "build a house"])

    assert result.exit_code == 0
    MockOrchestrator.assert_called_once()
    assert MockOrchestrator.call_args.kwargs["goal"] == "build a house"


def test_web_command_accepts_host_and_port():
    runner = CliRunner()
    with patch("furrow.cli.main.run") as mock_run:
        result = runner.invoke(
            main, ["web", "--host", "127.0.0.1", "--port", "9000"]
        )

    assert result.exit_code == 0
    mock_run.assert_called_once_with(host="127.0.0.1", port=9000)
