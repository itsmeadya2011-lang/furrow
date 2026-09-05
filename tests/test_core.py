import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_is_done_no_plan():
    mock_client = MagicMock()
    orchestrator = Orchestrator("test goal", client=mock_client)
    assert orchestrator._is_done() is False


def test_is_done_all_completed():
    mock_client = MagicMock()
    orchestrator = Orchestrator("test goal", client=mock_client)
    orchestrator._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


def test_is_done_some_pending():
    mock_client = MagicMock()
    orchestrator = Orchestrator("test goal", client=mock_client)
    orchestrator._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="pending"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_is_done_some_failed():
    mock_client = MagicMock()
    orchestrator = Orchestrator("test goal", client=mock_client)
    orchestrator._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="failed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_is_done_empty_tasks():
    mock_client = MagicMock()
    orchestrator = Orchestrator("test goal", client=mock_client)
    orchestrator._current_plan = Plan(tasks=[], rationale="ok")
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_goal_mutation_on_test_failure():
    mock_client = MagicMock()
    mock_client.settings = Settings(max_cycles=0)
    orchestrator = Orchestrator("Build a REST API", client=mock_client)

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1"),
        ],
        rationale="ok",
    )

    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan):
        with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"):
            with patch.object(
                TesterAgent,
                "run",
                new_callable=AsyncMock,
                return_value=TestResult(
                    passed=False,
                    summary="Tests failed",
                    failures=["test_a failed", "test_b failed"],
                ),
            ):
                await orchestrator._cycle()

    assert "Build a REST API" in orchestrator.goal
    assert "Fix failing tests:" in orchestrator.goal
    assert "test_a failed" in orchestrator.goal
    assert "test_b failed" in orchestrator.goal


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    settings = Settings(max_cycles=2)
    mock_client = MagicMock()
    mock_client.settings = settings

    orchestrator = Orchestrator("test goal", client=mock_client)

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task1"),
            TaskModel(id="2", description="task2"),
        ],
        rationale="ok",
    )

    async def worker_run_side_effect(self):
        if self.task.id == "1":
            return "done"
        raise RuntimeError("fail")

    passing_result = TestResult(passed=True, summary="ok", failures=[])

    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan):
        with patch.object(
            WorkerAgent, "run", new_callable=AsyncMock, side_effect=worker_run_side_effect
        ):
            with patch.object(
                TesterAgent, "run", new_callable=AsyncMock, return_value=passing_result
            ):
                await orchestrator.run()

    assert orchestrator.cycles == 2


@pytest.mark.asyncio
async def test_complete_ollama():
    settings = Settings(
        provider=Provider.OLLAMA,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
    )
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"message": {"content": "Hello from Ollama"}}

    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_async_client):
        result = await client._complete_ollama("Hello", "system", "llama3")

    assert result == "Hello from Ollama"
    mock_async_client.post.assert_called_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )


@pytest.mark.asyncio
async def test_run_tests_python_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    settings = Settings(workspace=tmp_path)
    agent = TesterAgent(settings=settings)

    calls = []

    async def mock_create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        proc = AsyncMock()
        proc.communicate.return_value = (b"passed", b"")
        proc.returncode = 0
        return proc

    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        side_effect=mock_create_subprocess_exec,
    ):
        output = await agent._run_tests()

    assert calls[0] == ("pytest", "-q")
    assert output == "passed"


@pytest.mark.asyncio
async def test_run_tests_node_project(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "test"}')

    settings = Settings(workspace=tmp_path)
    agent = TesterAgent(settings=settings)

    calls = []

    async def mock_create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        proc = AsyncMock()
        proc.communicate.return_value = (b"passed", b"")
        proc.returncode = 0
        return proc

    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        side_effect=mock_create_subprocess_exec,
    ):
        output = await agent._run_tests()

    assert calls[0] == ("npm", "test", "--", "--silent")
    assert output == "passed"


@pytest.mark.asyncio
async def test_run_tests_rust_project(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'\n")

    settings = Settings(workspace=tmp_path)
    agent = TesterAgent(settings=settings)

    calls = []

    async def mock_create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        proc = AsyncMock()
        proc.communicate.return_value = (b"passed", b"")
        proc.returncode = 0
        return proc

    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        side_effect=mock_create_subprocess_exec,
    ):
        output = await agent._run_tests()

    assert calls[0] == ("cargo", "test", "-q")
    assert output == "passed"


@pytest.mark.asyncio
async def test_run_tests_go_project(tmp_path):
    (tmp_path / "go.mod").write_text("module test\n")

    settings = Settings(workspace=tmp_path)
    agent = TesterAgent(settings=settings)

    calls = []

    async def mock_create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        proc = AsyncMock()
        proc.communicate.return_value = (b"passed", b"")
        proc.returncode = 0
        return proc

    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        side_effect=mock_create_subprocess_exec,
    ):
        output = await agent._run_tests()

    assert calls[0] == ("go", "test", "./...")
    assert output == "passed"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.anthropic_api_key is None
    assert s.openai_api_key is None
    assert s.ollama_api_key is None
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "llama3"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
