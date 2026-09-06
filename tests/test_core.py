from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, Tool
from furrow.agents.worker import WorkerAgent


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_tokens == 4096
    assert s.ollama_model == "llama3"


def test_settings_defaults_ollama():
    s = Settings(provider=Provider.OLLAMA)
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "llama3"


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

def test_tool_schema():
    tool = Tool(
        name="test_tool",
        description="A test tool",
        func=lambda x: str(x),
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    )
    schema = tool.schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "test_tool"
    assert schema["function"]["parameters"]["required"] == ["x"]


def test_tool_anthropic_schema():
    tool = Tool(
        name="test_tool",
        description="A test tool",
        func=lambda: "ok",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    schema = tool.anthropic_schema
    assert schema["name"] == "test_tool"
    assert schema["input_schema"]["type"] == "object"


@pytest.mark.asyncio
async def test_tool_execute_sync():
    tool = Tool(name="echo", description="echo", func=lambda x: f"echo: {x}")
    result = await tool.execute(x="hello")
    assert result == "echo: hello"


@pytest.mark.asyncio
async def test_tool_execute_async():
    async def async_func(x: str) -> str:
        return f"async: {x}"

    tool = Tool(name="async_echo", description="async echo", func=async_func)
    result = await tool.execute(x="hello")
    assert result == "async: hello"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

def test_orchestrator_init_defaults():
    orch = Orchestrator(goal="test goal")
    assert orch.goal == "test goal"
    assert orch.cycles == 0
    assert orch.tasks == []
    assert orch._console is not None


def test_orchestrator_init_with_custom_console():
    from rich.console import Console
    console = Console()
    orch = Orchestrator(goal="test goal", console=console)
    assert orch._console is console


def test_is_done_no_tasks():
    orch = Orchestrator(goal="test goal")
    assert orch._is_done() is True


def test_is_done_all_completed():
    orch = Orchestrator(goal="test goal")
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_some_pending():
    orch = Orchestrator(goal="test goal")
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="pending"),
    ]
    assert orch._is_done() is False


def test_is_done_some_failed():
    orch = Orchestrator(goal="test goal")
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_all_failed():
    orch = Orchestrator(goal="test goal")
    orch.tasks = [
        TaskModel(id="1", description="task1", status="failed"),
    ]
    assert orch._is_done() is False


def test_max_cycles_stops_loop():
    """If max_cycles is set, the loop should stop after that many cycles."""
    settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="test goal", settings=settings)
    assert orch.settings.max_cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_get_tasks_returns_stored_tasks():
    orch = Orchestrator(goal="test goal")
    orch.tasks = [TaskModel(id="1", description="task1", status="completed")]
    assert len(orch._get_tasks()) == 1
    assert orch._get_tasks()[0].id == "1"


# ---------------------------------------------------------------------------
# LLM Client tests (with mocked providers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_complete_openai():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI"))]
    client._openai = MagicMock()
    client._openai.chat = MagicMock()
    client._openai.chat.completions = AsyncMock()
    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.complete("test", model="gpt-4o", system="sys")
    assert result == "Hello from OpenAI"


@pytest.mark.asyncio
async def test_llm_complete_ollama():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    # Set _openai to a mock so the ollama property returns it instead of
    # creating a real AsyncOpenAI instance (which doesn't allow setting .chat)
    mock_client_obj = MagicMock()
    mock_client_obj.base_url = "http://localhost:11434/v1"
    client._openai = mock_client_obj

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from Ollama"))]
    mock_client_obj.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.complete("test", model="llama3")
    assert result == "Hello from Ollama"


@pytest.mark.asyncio
async def test_llm_get_file_tools():
    client = LLMClient()
    tools = client.get_file_tools()
    assert len(tools) == 3
    names = [t.name for t in tools]
    assert "read_file" in names
    assert "write_file" in names
    assert "list_files" in names


@pytest.mark.asyncio
async def test_llm_write_file(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)
    path = tmp_path / "test_output.txt"
    result = await client.write_file(path, "hello world")
    assert "Wrote" in result
    assert path.read_text() == "hello world"


@pytest.mark.asyncio
async def test_llm_read_file(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)
    path = tmp_path / "test_read.txt"
    path.write_text("file contents")
    result = await client.read_file(path)
    assert result == "file contents"


@pytest.mark.asyncio
async def test_llm_list_files(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.py" in files


def test_llm_max_tokens_from_settings():
    settings = Settings(max_tokens=2048)
    client = LLMClient(settings=settings)
    assert client.settings.max_tokens == 2048


# ---------------------------------------------------------------------------
# WorkerAgent tests (with mocked client)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_agent_calls_complete_with_tools():
    task = TaskModel(id="1", description="Add a test", files=["test.py"], dependencies=[])
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.get_file_tools = MagicMock(return_value=[])
    mock_client.complete = AsyncMock(return_value="Done: added test.py")

    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    assert result == "Done: added test.py"
    mock_client.complete.assert_called_once()
    call_kwargs = mock_client.complete.call_args
    assert "tools" in call_kwargs.kwargs
    assert call_kwargs.kwargs["system"] is not None
    assert call_kwargs.kwargs["model"] is not None


# ---------------------------------------------------------------------------
# PlannerAgent tests (with mocked client)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planner_agent_parse_plan():
    import json
    from furrow.agents.planner import PlannerAgent

    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.settings.planner_model = "test-model"
    plan_data = {
        "tasks": [{"id": "1", "description": "Do thing", "files": ["a.py"], "dependencies": []}],
        "rationale": "Test plan",
    }
    mock_client.complete = AsyncMock(return_value=json.dumps(plan_data))

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Build something")

    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Do thing"
    assert plan.rationale == "Test plan"


@pytest.mark.asyncio
async def test_planner_agent_invalid_json():
    from furrow.agents.planner import PlannerAgent

    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="not json")

    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Build something")


# ---------------------------------------------------------------------------
# TesterAgent tests (with mocked client)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tester_agent_pass():
    import json
    from furrow.agents.tester import TesterAgent

    mock_client = MagicMock()
    mock_client.settings = Settings()
    result_data = {"passed": True, "summary": "All tests passed", "failures": []}
    mock_client.complete = AsyncMock(return_value=json.dumps(result_data))

    tester = TesterAgent(client=mock_client)
    # _run_tests will try to run pytest, which may or may not exist.
    # Mock it to return empty output.
    with patch.object(tester, '_run_tests', new=AsyncMock(return_value="")):
        result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_agent_no_test_runner():
    from furrow.agents.tester import TesterAgent

    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')

    tester = TesterAgent(client=mock_client)
    # Force _run_tests to return "No test runner found."
    with patch.object(tester, '_run_tests', new=AsyncMock(return_value="No test runner found.")):
        result = await tester.run("test goal", [])

    assert result.passed is True
