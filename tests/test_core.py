from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import Message, TextBlock

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_provider_enum_values() -> None:
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"
    assert list(Provider) == [Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA]


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="do something")
    assert task.id == "1"
    assert task.description == "do something"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_plan_creation() -> None:
    tasks = [TaskModel(id="1", description="task one"), TaskModel(id="2", description="task two")]
    plan = Plan(tasks=tasks, rationale="because")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "because"


def test_test_result_defaults() -> None:
    result = TestResult(passed=True, summary="all good")
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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
    assert settings.test_timeout == 120
    assert settings.llm_retry_attempts == 3
    assert settings.llm_retry_backoff == 1.0


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4o")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    monkeypatch.setenv("FURROW_TEST_TIMEOUT", "60")

    settings = Settings()
    assert settings.provider == Provider.OPENAI
    assert settings.model == "gpt-4o"
    assert settings.max_cycles == 10
    assert settings.test_timeout == 60


@pytest.mark.asyncio
async def test_llm_complete_routes_to_anthropic() -> None:
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=settings)

    mock_response = Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="hello")],
        model="claude-3-5-sonnet-20241022",
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock, return_value="hello") as mock_complete:
        result = await client.complete("say hello")
        assert result == "hello"
        mock_complete.assert_called_once()


@pytest.mark.asyncio
async def test_llm_complete_routes_to_openai() -> None:
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    with patch.object(client, "_complete_openai", new_callable=AsyncMock, return_value="world") as mock_complete:
        result = await client.complete("say world")
        assert result == "world"
        mock_complete.assert_called_once()


@pytest.mark.asyncio
async def test_llm_complete_unsupported_provider() -> None:
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test")


@pytest.mark.asyncio
async def test_llm_missing_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(settings=settings)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


@pytest.mark.asyncio
async def test_llm_missing_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(provider=Provider.OPENAI, openai_api_key=None)
    client = LLMClient(settings=settings)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        _ = client.openai


@pytest.mark.asyncio
async def test_llm_retry_succeeds_on_second_attempt() -> None:
    settings = Settings(llm_retry_attempts=3, llm_retry_backoff=0.01)
    client = LLMClient(settings=settings)

    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("transient error")
        return "success"

    result = await client._retry(flaky)
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_llm_retry_exhausts() -> None:
    settings = Settings(llm_retry_attempts=2, llm_retry_backoff=0.01)
    client = LLMClient(settings=settings)

    async def always_fail():
        raise RuntimeError("permanent error")

    with pytest.raises(RuntimeError, match="permanent error"):
        await client._retry(always_fail)


@pytest.mark.asyncio
async def test_llm_read_file(tmp_path: Path) -> None:
    target = tmp_path / "input.txt"
    target.write_text("file contents", encoding="utf-8")

    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)

    result = await client.read_file(str(target))
    assert result == "file contents"


@pytest.mark.asyncio
async def test_llm_write_file_creates_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "output.txt"
    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)

    await client.write_file(str(target), "hello world")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello world"


def test_llm_list_files(tmp_path: Path) -> None:
    (tmp_path / "foo.txt").write_text("foo")
    (tmp_path / "bar.py").write_text("bar")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "baz.txt").write_text("baz")

    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)

    files = client.list_files(tmp_path)
    assert set(files) == {"foo.txt", "bar.py", "sub/baz.txt"}


def test_llm_list_files_missing_dir() -> None:
    settings = Settings(workspace=Path("/nonexistent/path/12345"))
    client = LLMClient(settings=settings)

    files = client.list_files(Path("/nonexistent/path/12345"))
    assert files == []


@pytest.mark.asyncio
async def test_planner_valid_json() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "tasks": [{"id": "1", "description": "build api", "files": ["api.py"], "dependencies": []}],
        "rationale": "api first",
    }))
    mock_client.settings = Settings(planner_model="test-model")

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("build a rest api")

    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].files == ["api.py"]
    assert plan.rationale == "api first"


@pytest.mark.asyncio
async def test_planner_markdown_wrapped_json() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="Here is the plan:\n```json\n{\n  \"tasks\": [],\n  \"rationale\": \"empty\"\n}\n```")
    mock_client.settings = Settings(planner_model="test-model")

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("plan nothing")

    assert len(plan.tasks) == 0
    assert plan.rationale == "empty"


@pytest.mark.asyncio
async def test_planner_invalid_json_raises() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="this is not json")
    mock_client.settings = Settings(planner_model="test-model")

    planner = PlannerAgent(client=mock_client)

    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await planner.plan("do something")


@pytest.mark.asyncio
async def test_worker_code_block_extraction(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=(
        "Here is the implementation:\n"
        "# File: hello.py\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
        "Summary done."
    ))
    mock_client.settings = Settings(workspace=tmp_path, worker_model="test-model")

    task = TaskModel(id="1", description="say hello", files=["hello.py"])
    worker = WorkerAgent(task=task, client=mock_client)

    summary = await worker.run()

    target = tmp_path / "hello.py"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "print('hello')"
    assert "Wrote files" in summary


@pytest.mark.asyncio
async def test_worker_no_code_blocks_returns_raw() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="just text, no code blocks")
    mock_client.settings = Settings(worker_model="test-model")

    task = TaskModel(id="1", description="do something")
    worker = WorkerAgent(task=task, client=mock_client)

    result = await worker.run()
    assert result == "just text, no code blocks"


@pytest.mark.asyncio
async def test_worker_skips_blocks_without_file_comment() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=(
        "```python\n"
        "x = 1\n"
        "```"
    ))
    mock_client.settings = Settings(workspace=Path.cwd(), worker_model="test-model")

    task = TaskModel(id="1", description="do something")
    worker = WorkerAgent(task=task, client=mock_client)

    result = await worker.run()
    assert result == "```python\nx = 1\n```"


@pytest.mark.asyncio
async def test_tester_passed_result() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": True,
        "summary": "all tests passed",
        "failures": [],
    }))
    mock_client.settings = Settings(tester_model="test-model")

    with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock, return_value="test output"):
        tester = TesterAgent(client=mock_client)
        result = await tester.run("build api", [])
        assert result.passed is True
        assert result.summary == "all tests passed"


@pytest.mark.asyncio
async def test_tester_failed_result() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": False,
        "summary": "tests failed",
        "failures": ["test_a failed"],
    }))
    mock_client.settings = Settings(tester_model="test-model")

    with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock, return_value="test output"):
        tester = TesterAgent(client=mock_client)
        result = await tester.run("build api", [])
        assert result.passed is False
        assert result.failures == ["test_a failed"]


@pytest.mark.asyncio
async def test_tester_invalid_json_falls_back_to_keyword_detection() -> None:
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="All tests passed successfully")
    mock_client.settings = Settings(tester_model="test-model")

    with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock, return_value="test output"):
        tester = TesterAgent(client=mock_client)
        result = await tester.run("build api", [])
        assert result.passed is True


@pytest.mark.asyncio
async def test_tester_run_tests_timeout_continues(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, test_timeout=0)

    mock_client = MagicMock()
    mock_client.settings = settings
    mock_client.complete = AsyncMock()

    tester = TesterAgent(client=mock_client)

    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError("no test runner")

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        output = await tester._run_tests()
        assert "No test runner found" in output


@pytest.mark.asyncio
async def test_orchestrator_get_tasks() -> None:
    orchestrator = Orchestrator("test goal")
    orchestrator.last_plan = Plan(
        tasks=[TaskModel(id="1", description="t1"), TaskModel(id="2", description="t2")],
        rationale="ok",
    )

    tasks = orchestrator._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_completed_tasks() -> None:
    orchestrator = Orchestrator("test goal")
    orchestrator.last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ],
        rationale="ok",
    )

    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failed_tasks() -> None:
    orchestrator = Orchestrator("test goal")
    orchestrator.last_plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ],
        rationale="ok",
    )

    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_pending_tasks() -> None:
    orchestrator = Orchestrator("test goal")
    orchestrator.last_plan = Plan(
        tasks=[TaskModel(id="1", description="t1", status="pending")],
        rationale="ok",
    )

    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_max_cycles() -> None:
    orchestrator = Orchestrator("test goal")
    orchestrator.cycles = 5
    orchestrator.settings.max_cycles = 5

    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_no_plan() -> None:
    orchestrator = Orchestrator("test goal")
    orchestrator.last_plan = None

    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_status_callback_called() -> None:
    mock_client = MagicMock()
    mock_plan = Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok")
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=mock_plan)
    mock_tester = MagicMock()
    mock_tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    with patch("furrow.core.orchestrator.PlannerAgent", return_value=mock_planner):
        with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
            orchestrator = Orchestrator("test goal", client=mock_client)
            orchestrator.last_plan = Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok")

            callback = MagicMock()
            orchestrator.status_callback = callback
            await orchestrator._cycle()

            calls = [call.args[0] for call in callback.call_args_list]
            assert any("Planning..." in str(c) for c in calls)
            assert any("Executing" in str(c) for c in calls)
            assert any("Testing..." in str(c) for c in calls)
            assert any("Task 1" in str(c) for c in calls)
