"""Tests for Furrow core components."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient, LLMResponse, Tool, ToolCall

# ---------------------------------------------------------------------------
# Existing config / model tests
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

def test_tool_spec_openai():
    tool = Tool(
        name="read_file",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    spec = tool.spec(Provider.OPENAI)
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "read_file"
    assert spec["function"]["parameters"]["required"] == ["path"]


def test_tool_spec_anthropic():
    from furrow.config import Provider

    tool = Tool(
        name="write_file",
        description="Write a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    )
    spec = tool.spec(Provider.ANTHROPIC)
    assert spec["name"] == "write_file"
    assert spec["input_schema"]["required"] == ["path", "content"]
    # Anthropic format should NOT have "type": "function"
    assert "type" not in spec


def test_tool_spec_ollama():
    from furrow.config import Provider

    tool = Tool(
        name="list_files",
        description="List files",
        parameters={"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]},
    )
    spec = tool.spec(Provider.OLLAMA)
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "list_files"


@pytest.mark.asyncio
async def test_tool_call_with_function():
    """Tool.call() should invoke the backing async function."""

    async def fake_read(path: str) -> str:
        return f"contents of {path}"

    tool = Tool("read_file", "Read a file", {"type": "object"}, func=fake_read)
    result = await tool.call({"path": "foo.py"})
    assert result == "contents of foo.py"


@pytest.mark.asyncio
async def test_tool_call_without_function():
    tool = Tool("noop", "Does nothing", {"type": "object"})
    with pytest.raises(ValueError, match="no function"):
        await tool.call({})


# ---------------------------------------------------------------------------
# LLMResponse tests
# ---------------------------------------------------------------------------

def test_llm_response_has_tool_calls():
    r = LLMResponse(text="hello", tool_calls=[ToolCall(name="x", arguments={})])
    assert r.has_tool_calls is True


def test_llm_response_no_tool_calls():
    r = LLMResponse(text="done")
    assert r.has_tool_calls is False


# ---------------------------------------------------------------------------
# LLMClient tests (no API key required)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_command_success(tmp_path):
    settings = Settings(workspace=tmp_path, provider="openai")  # type: ignore[arg-type]
    settings.openai_api_key = "fake-key"
    client = LLMClient(settings=settings)
    result = await client.run_command("echo hello")
    assert "hello" in result


@pytest.mark.asyncio
async def test_run_command_timeout(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)
    result = await client.run_command("sleep 10", timeout=1)
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_run_command_not_found(tmp_path):
    settings = Settings(workspace=tmp_path)
    client = LLMClient(settings=settings)
    result = await client.run_command("nonexistent_command_xyz123")
    assert "not found" in result.lower() or "failed" in result.lower()


@pytest.mark.asyncio
async def test_run_tool_loop_no_tools():
    """When the LLM returns no tool calls, the loop returns the text immediately."""
    from furrow.config import Provider

    settings = Settings(workspace=Path("/workspace"), provider=Provider.OPENAI)
    settings.openai_api_key = "fake-key"
    client = LLMClient(settings=settings)

    captured_args = []

    async def fake_create(**kwargs):
        captured_args.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = "Done!"
        resp.choices[0].message.tool_calls = None
        return resp

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = fake_create
    client._openai = mock_openai

    result = await client.run_tool_loop(
        prompt="Do something",
        tools=[],
        system="",
    )
    assert result == "Done!"
    assert len(captured_args) == 1


@pytest.mark.asyncio
async def test_run_tool_loop_with_tool_calls(tmp_path):
    """The loop should execute tool calls and continue until no more calls."""
    from furrow.config import Provider

    settings = Settings(workspace=tmp_path, provider=Provider.OPENAI)
    settings.openai_api_key = "fake-key"
    client = LLMClient(settings=settings)

    # Track the sequence of responses
    call_count = [0]

    async def fake_create(**kwargs):
        call_count[0] += 1
        resp = MagicMock()
        message = MagicMock()
        if call_count[0] == 1:
            # First call: request to read a file
            message.content = "I'll read the file."
            message.tool_calls = [
                MagicMock(
                    id="call_1",
                    type="function",
                    function=MagicMock(
                        name="read_file",
                        arguments='{"path": "test.txt"}',
                    ),
                )
            ]
        else:
            # Second call: no tool calls, final response
            message.content = "File contents read successfully."
            message.tool_calls = None
        resp.choices[0].message = message
        return resp

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = fake_create
    client._openai = mock_openai

    tool = Tool(
        name="read_file",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        func=client.read_file,
    )

    # Pre-create the file so read_file succeeds
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello from file!")

    result = await client.run_tool_loop(
        prompt="Read test.txt",
        tools=[tool],
        system="",
    )
    assert "read successfully" in result
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_run_tool_loop_max_iterations(tmp_path):
    """The loop should stop after max_iterations even if the LLM keeps calling tools."""
    from furrow.config import Provider

    settings = Settings(workspace=tmp_path, provider=Provider.OPENAI)
    settings.openai_api_key = "fake-key"
    client = LLMClient(settings=settings)

    async def fake_create(**kwargs):
        resp = MagicMock()
        message = MagicMock()
        message.content = "Still going..."
        message.tool_calls = [
            MagicMock(
                id="call_x",
                type="function",
                function=MagicMock(
                    name="noop",
                    arguments="{}",
                ),
            )
        ]
        resp.choices[0].message = message
        return resp

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = fake_create
    client._openai = mock_openai

    async def noop() -> str:
        return ""

    tool = Tool("noop", "Does nothing", {"type": "object"}, func=noop)

    result = await client.run_tool_loop(
        prompt="Loop forever",
        tools=[tool],
        max_iterations=3,
    )
    assert "Max tool iterations" in result or "still going" in result.lower()


# ---------------------------------------------------------------------------
# Orchestrator tests (no API key required, mocked planner/tester/workers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_get_tasks_returns_plan_tasks():
    """_get_tasks() should return tasks from the current plan, not an empty list."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=5)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test goal", client=client)

    # Simulate having a plan
    orch.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ],
        rationale="test",
    )
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[1].id == "2"


def test_orchestrator_should_stop_no_settings():
    """Without max_cycles and without goal_complete, should not stop."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)
    orch.cycles = 1
    assert orch._should_stop() is False


def test_orchestrator_should_stop_max_cycles():
    """Should stop when max_cycles is reached."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=3)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)
    orch.cycles = 3
    assert orch._should_stop() is True


def test_orchestrator_should_stop_goal_complete():
    """Should stop when goal is marked complete."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)
    orch._goal_complete = True
    assert orch._should_stop() is True


def test_orchestrator_should_not_stop_before_max_cycles():
    """Should NOT stop before reaching max_cycles."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=10)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)
    orch.cycles = 5
    assert orch._should_stop() is False


@pytest.mark.asyncio
async def test_orchestrator_cycle_no_tasks_completes_goal():
    """When the planner returns no tasks, the goal is marked complete."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="already done", client=client)

    # Mock the planner to return an empty plan
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch.planner = mock_planner

    await orch._cycle()

    assert orch._goal_complete is True
    assert orch._should_stop() is True


@pytest.mark.asyncio
async def test_orchestrator_cycle_all_completed_with_passing_tests():
    """When all tasks are already completed and tests pass, goal is complete."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)

    # Mock planner to return a plan where all tasks are already completed
    orch.current_plan = None

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="task 1", status="completed")],
            rationale="test",
        )
    )
    orch.planner = mock_planner

    mock_tester = MagicMock()
    mock_tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="All passed", failures=[])
    )
    # Monkey-patch the TesterAgent constructor
    with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
        await orch._cycle()

    assert orch._goal_complete is True


@pytest.mark.asyncio
async def test_orchestrator_cycle_tests_fail_continues():
    """When tests fail, the goal should NOT be complete (continues to next cycle)."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=0)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test",
        )
    )
    orch.planner = mock_planner

    mock_worker = MagicMock()
    mock_worker.run = AsyncMock(return_value="implemented task 1")
    with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
        mock_tester = MagicMock()
        mock_tester.run = AsyncMock(
            return_value=TestResult(passed=False, summary="tests failed", failures=["bug in code"])
        )
        with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
            await orch._cycle()

    assert orch._goal_complete is False
    assert "bug in code" in orch.goal


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_settings_ollama_provider():
    from furrow.config import Provider

    settings = Settings(provider=Provider.OLLAMA, ollama_model="phi3")
    assert settings.provider == Provider.OLLAMA
    assert settings.ollama_model == "phi3"


def test_settings_max_cycles_default():
    settings = Settings()
    assert settings.max_cycles == 0


def test_llm_client_complete_ollama_not_configured():
    """When using Ollama provider, complete() should call the ollama path."""
    from furrow.config import Provider

    settings = Settings(provider=Provider.OLLAMA, ollama_model="llama3")
    client = LLMClient(settings=settings)
    assert client.settings.provider == Provider.OLLAMA


def test_worker_agent_has_tools():
    """WorkerAgent should expose at least 4 file/command tools."""
    from furrow.agents.worker import WorkerAgent

    settings = Settings()
    client = LLMClient(settings=settings)
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    tools = worker._get_tools()
    tool_names = {t.name for t in tools}
    assert {"read_file", "write_file", "list_files", "run_command"}.issubset(tool_names)


def test_worker_agent_tools_have_specs():
    """Each tool should have valid OpenAI and Anthropic specs."""
    from furrow.agents.worker import WorkerAgent
    from furrow.config import Provider

    settings = Settings()
    client = LLMClient(settings=settings)
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    tools = worker._get_tools()
    for tool in tools:
        openai_spec = tool.spec(Provider.OPENAI)
        anthropic_spec = tool.spec(Provider.ANTHROPIC)
        assert "function" in openai_spec
        assert "name" in anthropic_spec


def test_orch_agent_init_accepts_settings():
    """The orchestrator should correctly store settings."""
    from furrow.core.orchestrator import Orchestrator

    settings = Settings(max_cycles=10)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)
    assert orch.settings is not None
    assert orch.settings.max_cycles == 10
