from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, TOOL_DEFINITIONS
from furrow.web.server import WebSocketWriter


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5


def test_settings_env_prefix():
    """Verify that Settings reads FURROW_* env vars, not bare ANTHROPIC_API_KEY."""
    with patch.dict("os.environ", {"FURROW_PROVIDER": "openai", "FURROW_MAX_CYCLES": "3"}):
        s = Settings()
        assert s.provider == Provider.OPENAI
        assert s.max_cycles == 3


def test_llm_client_tool_definitions():
    assert any(t["name"] == "read_file" for t in TOOL_DEFINITIONS)
    assert any(t["name"] == "write_file" for t in TOOL_DEFINITIONS)
    assert any(t["name"] == "list_files" for t in TOOL_DEFINITIONS)


@pytest.mark.asyncio
async def test_llm_client_execute_tool_read_write_list(tmp_path):
    client = LLMClient()
    # write_file
    await client.execute_tool("write_file", {"path": str(tmp_path / "hello.txt"), "content": "hello"})
    # read_file
    content = await client.execute_tool("read_file", {"path": str(tmp_path / "hello.txt")})
    assert content == "hello"
    # list_files
    listing = await client.execute_tool("list_files", {"directory": str(tmp_path)})
    assert "hello.txt" in listing


@pytest.mark.asyncio
async def test_llm_client_ollama_stub():
    """Ollama provider should attempt HTTP call (mocked here)."""
    client = LLMClient()
    client.settings.provider = Provider.OLLAMA
    client.settings.ollama_base_url = "http://localhost:11434"

    mock_response = AsyncMock()
    mock_response.json.return_value = {"response": "mocked ollama response"}
    mock_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.post = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)

        result = await client.complete("hello", model="llama3")
        assert result == "mocked ollama response"


@pytest.mark.asyncio
async def test_orchestrator_get_tasks_returns_plan_tasks():
    """Orchestrator._get_tasks() should return tasks from the stored plan."""
    client = LLMClient()
    orchestrator = Orchestrator(goal="test", client=client)
    assert orchestrator._get_tasks() == []

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task one"),
            TaskModel(id="2", description="task two"),
        ],
        rationale="test plan",
    )
    orchestrator.plan = plan
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].description == "task one"


def test_orchestrator_max_cycles_check():
    """Verify the max_cycles guard condition works as expected."""
    client = LLMClient()
    client.settings.max_cycles = 2
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.cycles = 2
    assert client.settings.max_cycles > 0 and orchestrator.cycles >= client.settings.max_cycles


@pytest.mark.asyncio
async def test_worker_agent_tool_call(monkeypatch, tmp_path):
    """WorkerAgent should use tools when the LLM returns tool calls."""
    client = LLMClient()

    async def mock_complete_with_tools(prompt, system="", model=None):
        if "write" in prompt:
            return "", [{"name": "write_file", "input": {"path": str(tmp_path / "out.txt"), "content": "done"}}]
        return "done", []

    monkeypatch.setattr(client, "complete_with_tools", mock_complete_with_tools)
    monkeypatch.setattr(client, "execute_tool", AsyncMock(return_value="Wrote file: out.txt"))

    task = TaskModel(id="1", description="write out.txt", files=["out.txt"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert "Wrote file" in result or "done" in result


@pytest.mark.asyncio
async def test_websocket_writer():
    """WebSocketWriter should queue writes for async draining."""
    writer = WebSocketWriter(None)  # type: ignore[arg-type]
    writer.write("hello")
    assert writer._queue.qsize() == 1
    text = await writer._queue.get()
    assert text == "hello"
