import asyncio
from pathlib import Path

import pytest

from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="2", description="another")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_test_result_failures_default():
    t = TestResult(passed=False, summary="broken")
    assert t.failures == []


# --- Orchestrator state management tests ---

class _FakeTask:
    """Minimal stand-in for a TaskModel to test _is_done logic."""

    def __init__(self, status: str = "completed"):
        self.status = status
        self.id = "1"


def test_orchestrator_done_when_no_tasks():
    from furrow.core.orchestrator import Orchestrator

    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_not_done_when_failed():
    from furrow.core.orchestrator import Orchestrator

    o = Orchestrator(goal="test")
    o.tasks = [_FakeTask(status="failed")]
    assert o._is_done() is False


def test_orchestrator_not_done_when_pending():
    from furrow.core.orchestrator import Orchestrator

    o = Orchestrator(goal="test")
    o.tasks = [_FakeTask(status="pending")]
    assert o._is_done() is False


def test_orchestrator_done_when_all_completed():
    from furrow.core.orchestrator import Orchestrator

    o = Orchestrator(goal="test")
    o.tasks = [_FakeTask(status="completed"), _FakeTask(status="completed")]
    assert o._is_done() is True


def test_orchestrator_not_done_when_mixed():
    from furrow.core.orchestrator import Orchestrator

    o = Orchestrator(goal="test")
    o.tasks = [_FakeTask(status="completed"), _FakeTask(status="pending")]
    assert o._is_done() is False


# --- LLMClient tool methods tests ---

@pytest.mark.asyncio
async def test_llm_client_read_write_file(tmp_path):
    from furrow.llm import LLMClient

    client = LLMClient()
    path = tmp_path / "hello.txt"
    await client.write_file(path, "hello world")
    content = await client.read_file(path)
    assert content == "hello world"


@pytest.mark.asyncio
async def test_llm_client_write_creates_dirs(tmp_path):
    from furrow.llm import LLMClient

    client = LLMClient()
    path = tmp_path / "a" / "b" / "c.txt"
    await client.write_file(path, "deep")
    assert path.read_text() == "deep"


def test_llm_client_list_files(tmp_path):
    from furrow.llm import LLMClient

    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("z")

    client = LLMClient()
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.py" in files
    assert str(Path("sub/c.py")) in files or "sub/c.py" in files


def test_llm_client_list_files_missing_dir():
    from furrow.llm import LLMClient

    client = LLMClient()
    assert client.list_files("/nonexistent/path/xyz") == []


@pytest.mark.asyncio
async def test_llm_client_run_command_success():
    from furrow.llm import LLMClient

    client = LLMClient()
    result = await client.run_command("echo hello")
    assert "hello" in result


@pytest.mark.asyncio
async def test_llm_client_run_command_failure():
    from furrow.llm import LLMClient

    client = LLMClient()
    result = await client.run_command("exit 1")
    assert "exit code: 1" in result


# --- Worker agent tool definitions tests ---

def test_worker_tool_definitions_exist():
    from furrow.agents.worker import ANTHROPIC_TOOLS, OPENAI_TOOLS

    names = {t["name"] for t in ANTHROPIC_TOOLS}
    assert {"read_file", "write_file", "run_command", "list_files"} <= names

    openai_names = {t["function"]["name"] for t in OPENAI_TOOLS}
    assert openai_names == names


def test_worker_openai_tools_have_function_wrapper():
    from furrow.agents.worker import OPENAI_TOOLS

    for tool in OPENAI_TOOLS:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "parameters" in tool["function"]


# --- Ollama provider tests ---

def test_ollama_provider_config():
    from furrow.config import Provider, Settings

    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    assert s.provider == Provider.OLLAMA


def test_ollama_client_lazy_init():
    from furrow.config import Provider, Settings
    from furrow.llm import LLMClient

    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)
    assert client._ollama is None
    # Accessing the property should create the client
    ollama = client.ollama
    assert ollama is not None
    assert client._ollama is ollama


# --- Orchestrator event callback tests ---

@pytest.mark.asyncio
async def test_orchestrator_emit_calls_callback():
    from furrow.core.orchestrator import Orchestrator

    events = []

    def on_event(msg: str):
        events.append(msg)

    o = Orchestrator(goal="test", on_event=on_event)
    o._emit("hello")
    assert "hello" in events[-1]


@pytest.mark.asyncio
async def test_orchestrator_emit_async_callback():
    from furrow.core.orchestrator import Orchestrator

    events = []

    async def on_event(msg: str):
        events.append(msg)

    o = Orchestrator(goal="test", on_event=on_event)
    o._emit("async hello")
    # Give the scheduled coroutine a chance to run
    await asyncio.sleep(0.01)
    assert any("async hello" in e for e in events)


# --- Config tests ---

def test_settings_defaults():
    from furrow.config import Provider, Settings

    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_env_override(monkeypatch):
    from furrow.config import Settings

    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    s = Settings()
    assert s.provider.value == "openai"
    assert s.max_cycles == 10