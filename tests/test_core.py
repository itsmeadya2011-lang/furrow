"""Tests for furrow configuration and models."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, parse_json_response

# ---------------------------------------------------------------------------
# Existing tests (model construction)
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_response_plain():
    response = '{"passed": true, "summary": "ok"}'
    assert parse_json_response(response) == '{"passed": true, "summary": "ok"}'


def test_parse_json_response_markdown_fence_with_lang():
    response = '```json\n{"passed": true}\n```'
    assert parse_json_response(response) == '{"passed": true}'


def test_parse_json_response_markdown_fence_no_lang():
    response = '```\n{"passed": false}\n```'
    assert parse_json_response(response) == '{"passed": false}'


def test_parse_json_response_embedded_in_prose():
    response = 'Here is the result:\n{"passed": true, "summary": "ok"}\nDone.'
    result = parse_json_response(response)
    assert json.loads(result) == {"passed": True, "summary": "ok"}


def test_parse_json_response_empty():
    assert parse_json_response("") == ""


# ---------------------------------------------------------------------------
# TaskModel
# ---------------------------------------------------------------------------


def test_task_model_defaults():
    t = TaskModel(id="1", description="test task")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_task_model_with_optional_fields():
    t = TaskModel(
        id="2",
        description="do something",
        files=["src/app.py"],
        dependencies=["1"],
        status="completed",
        result="Done",
    )
    assert t.id == "2"
    assert t.files == ["src/app.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"


def test_task_model_status_transitions():
    """TaskModel should allow status updates."""
    t = TaskModel(id="1", description="task")
    assert t.status == "pending"
    t.status = "completed"
    assert t.status == "completed"
    t.status = "failed"
    assert t.status == "failed"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_tokens == 4096
    assert s.log_level == "INFO"


def test_settings_field_overrides():
    s = Settings(provider=Provider.OPENAI, max_cycles=3, model="gpt-4")
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 3
    assert s.model == "gpt-4"


def test_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    monkeypatch.setenv("FURROW_MODEL", "llama3")
    s = Settings()
    assert s.provider == Provider.OLLAMA
    assert s.max_cycles == 10
    assert s.model == "llama3"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_client_write_file_creates_dirs(tmp_path):
    client = LLMClient()
    target = tmp_path / "subdir" / "file.txt"
    await client.write_file(target, "hello world")
    assert target.read_text() == "hello world"


@pytest.mark.asyncio
async def test_llm_client_read_file(tmp_path):
    client = LLMClient()
    f = tmp_path / "test.txt"
    f.write_text("content here")
    result = await client.read_file(f)
    assert result == "content here"


@pytest.mark.asyncio
async def test_llm_client_read_file_not_found(tmp_path):
    client = LLMClient()
    with pytest.raises(FileNotFoundError):
        await client.read_file(tmp_path / "nonexistent.txt")


def test_llm_client_list_files(tmp_path):
    client = LLMClient()
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("c")
    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "b.txt" in files
    assert str(Path("sub") / "c.txt") in files


def test_llm_client_list_files_nonexistent():
    client = LLMClient()
    assert client.list_files("/nonexistent/path/xyz") == []


def test_llm_client_ollama_configured():
    """Ollama provider should be a valid, routable provider."""
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OLLAMA


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


def _make_mock_client(settings=None):
    """Create a mock LLMClient with the given settings."""
    mock = MagicMock(spec=LLMClient)
    mock.settings = settings or Settings()
    mock.complete = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_planner_agent_parse_plan():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = json.dumps(
        {
            "tasks": [{"id": "1", "description": "task1", "files": [], "dependencies": []}],
            "rationale": "test",
        }
    )
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Add a feature")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "task1"
    assert plan.rationale == "test"


@pytest.mark.asyncio
async def test_planner_agent_parse_markdown_response():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = '```json\n{"tasks": [], "rationale": "done"}\n```'
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Already done")
    assert len(plan.tasks) == 0
    assert plan.rationale == "done"


@pytest.mark.asyncio
async def test_planner_agent_invalid_json_raises():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = "Not valid JSON at all"
    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Some goal")


@pytest.mark.asyncio
async def test_planner_agent_passes_system_prompt():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = '{"tasks": [], "rationale": "ok"}'
    planner = PlannerAgent(client=mock_client)
    await planner.plan("test goal")
    call_kwargs = mock_client.complete.call_args
    assert call_kwargs is not None
    assert "system" in call_kwargs.kwargs
    assert "planning agent" in call_kwargs.kwargs["system"].lower()


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_agent_returns_result():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = "Implemented feature X"
    task = TaskModel(id="1", description="Implement feature X", files=["src/feature.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    assert result == "Implemented feature X"


@pytest.mark.asyncio
async def test_worker_agent_passes_system_prompt():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = "done"
    task = TaskModel(id="1", description="test", files=["src/app.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    await worker.run()
    call_kwargs = mock_client.complete.call_args
    assert "system" in call_kwargs.kwargs
    assert "worker agent" in call_kwargs.kwargs["system"].lower()


# ---------------------------------------------------------------------------
# TesterAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tester_agent_parse_passed():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = json.dumps(
        {"passed": True, "summary": "All tests passed", "failures": []}
    )
    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="1 passed"):
        result = await tester.run("Some goal", [])
    assert result.passed is True
    assert result.summary == "All tests passed"


@pytest.mark.asyncio
async def test_tester_agent_parse_failed():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = json.dumps(
        {"passed": False, "summary": "2 failed", "failures": ["test_foo", "test_bar"]}
    )
    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="2 failed"):
        result = await tester.run("Some goal", [])
    assert result.passed is False
    assert len(result.failures) == 2
    assert "test_foo" in result.failures


@pytest.mark.asyncio
async def test_tester_agent_fallback_on_bad_json():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = "Tests passed but I can't format JSON"
    tester = TesterAgent(client=mock_client)
    with patch.object(
        tester, "_run_tests", new_callable=AsyncMock, return_value="all tests passed"
    ):
        result = await tester.run("Some goal", [])
    assert result.passed is True
    assert result.summary == "Tests passed but I can't format JSON"


@pytest.mark.asyncio
async def test_tester_agent_handles_test_exception():
    mock_client = _make_mock_client()
    tester = TesterAgent(client=mock_client)
    with patch.object(
        tester, "_run_tests", new_callable=AsyncMock, side_effect=OSError("No tests")
    ):
        result = await tester.run("Some goal", [])
    assert result.passed is False
    assert "No tests" in result.summary


@pytest.mark.asyncio
async def test_tester_agent_passes_system_prompt():
    mock_client = _make_mock_client()
    mock_client.complete.return_value = '{"passed": true, "summary": "ok", "failures": []}'
    tester = TesterAgent(client=mock_client)
    with patch.object(tester, "_run_tests", new_callable=AsyncMock, return_value="ok"):
        await tester.run("test goal", [])
    call_kwargs = mock_client.complete.call_args
    assert "system" in call_kwargs.kwargs
    assert "tester agent" in call_kwargs.kwargs["system"].lower()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _make_test_console():
    """Create a Console that writes to a StringIO buffer (no terminal output)."""
    import io

    return Console(
        file=io.StringIO(),
        force_terminal=False,
        no_color=True,
        width=120,
        highlight=False,
    )


def _make_orchestrator(settings=None, mock_client=None):
    """Create an Orchestrator with a no-op console and optional settings."""
    settings = settings or Settings()
    if mock_client is None:
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = settings
        mock_client.complete = AsyncMock()
    return Orchestrator(
        goal="test goal",
        client=mock_client,
        console=_make_test_console(),
    )


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    """Orchestrator should stop after reaching max_cycles."""
    settings = Settings(max_cycles=1)
    mock_client = _make_mock_client(settings=settings)

    orch = Orchestrator(goal="test goal", client=mock_client, console=_make_test_console())
    with (
        patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan,
        patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker,
        patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_test,
    ):
        mock_plan.return_value = Plan(
            tasks=[TaskModel(id="1", description="test")], rationale="test"
        )
        mock_worker.return_value = "done"
        mock_test.return_value = TestResult(passed=False, summary="fail", failures=["error"])
        await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_completes_on_success():
    """Orchestrator should stop when all tasks complete and tests pass."""
    settings = Settings()
    mock_client = _make_mock_client(settings=settings)

    orch = Orchestrator(goal="test goal", client=mock_client, console=_make_test_console())
    with (
        patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan,
        patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker,
        patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_test,
    ):
        mock_plan.return_value = Plan(
            tasks=[TaskModel(id="1", description="test")], rationale="test"
        )
        mock_worker.return_value = "done"
        mock_test.return_value = TestResult(passed=True, summary="all passed", failures=[])
        await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_continues_on_test_failure():
    """Orchestrator should continue when tests fail (up to max_cycles)."""
    settings = Settings(max_cycles=5)
    mock_client = _make_mock_client(settings=settings)

    orch = Orchestrator(goal="test goal", client=mock_client, console=_make_test_console())
    with (
        patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan,
        patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker,
        patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_test,
    ):
        mock_plan.return_value = Plan(
            tasks=[TaskModel(id="1", description="test")], rationale="test"
        )
        mock_worker.return_value = "done"
        mock_test.side_effect = [
            TestResult(passed=False, summary="fail", failures=["error1"]),
            TestResult(passed=True, summary="ok", failures=[]),
        ]
        await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles_with_failures():
    """Even with test failures, max_cycles should stop the loop."""
    settings = Settings(max_cycles=2)
    mock_client = _make_mock_client(settings=settings)

    orch = Orchestrator(goal="test goal", client=mock_client, console=_make_test_console())
    with (
        patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan,
        patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker,
        patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_test,
    ):
        mock_plan.return_value = Plan(
            tasks=[TaskModel(id="1", description="test")], rationale="test"
        )
        mock_worker.return_value = "done"
        mock_test.return_value = TestResult(passed=False, summary="fail", failures=["error"])
        await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_no_tasks_stops():
    """If the planner returns no tasks, the orchestrator should stop."""
    settings = Settings()
    mock_client = _make_mock_client(settings=settings)

    orch = Orchestrator(goal="test goal", client=mock_client, console=_make_test_console())
    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan:
        mock_plan.return_value = Plan(tasks=[], rationale="done")
        await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_uses_max_parallel_tasks():
    """Orchestrator should create a semaphore with max_parallel_tasks limit."""
    settings = Settings(max_parallel_tasks=3)
    mock_client = _make_mock_client(settings=settings)

    orch = Orchestrator(goal="test goal", client=mock_client, console=_make_test_console())
    with (
        patch.object(PlannerAgent, "plan", new_callable=AsyncMock) as mock_plan,
        patch.object(WorkerAgent, "run", new_callable=AsyncMock) as mock_worker,
        patch.object(TesterAgent, "run", new_callable=AsyncMock) as mock_test,
        patch("furrow.core.orchestrator.asyncio.Semaphore") as mock_sem,
    ):
        mock_plan.return_value = Plan(
            tasks=[TaskModel(id=str(i), description=f"task{i}") for i in range(5)],
            rationale="test",
        )
        mock_worker.return_value = "done"
        mock_test.return_value = TestResult(passed=True, summary="ok", failures=[])
        await orch.run()
    mock_sem.assert_called_once_with(3)


def test_orchestrator_is_done_with_no_tasks():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    orch = Orchestrator(goal="test", client=mock_client, console=_make_test_console())
    orch.tasks = []
    orch.test_result = None
    assert orch._is_done() is True


def test_orchestrator_is_done_not_complete():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    orch = Orchestrator(goal="test", client=mock_client, console=_make_test_console())
    orch.tasks = [TaskModel(id="1", description="test")]
    orch.test_result = None
    assert orch._is_done() is False


def test_orchestrator_is_done_with_failed_task():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    orch = Orchestrator(goal="test", client=mock_client, console=_make_test_console())
    task = TaskModel(id="1", description="test")
    task.status = "failed"
    orch.tasks = [task]
    orch.test_result = None
    assert orch._is_done() is False


def test_orchestrator_is_done_with_failed_tests():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    orch = Orchestrator(goal="test", client=mock_client, console=_make_test_console())
    task = TaskModel(id="1", description="test")
    task.status = "completed"
    orch.tasks = [task]
    orch.test_result = TestResult(passed=False, summary="fail", failures=["error"])
    assert orch._is_done() is False


def test_orchestrator_is_done_with_passed_tests():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    orch = Orchestrator(goal="test", client=mock_client, console=_make_test_console())
    task = TaskModel(id="1", description="test")
    task.status = "completed"
    orch.tasks = [task]
    orch.test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True
