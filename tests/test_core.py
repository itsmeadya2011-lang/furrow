import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult, Provider, settings
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def _make_orchestrator(tmp_path, mock_client=None, goal="goal"):
    client = mock_client or MagicMock()
    sp = tmp_path / "state.json"
    return Orchestrator(goal=goal, client=client, state_path=sp)


def test_is_done_all_completed(tmp_path):
    o = _make_orchestrator(tmp_path)
    o.tasks = [TaskModel(id="1", description="a", status="completed"), TaskModel(id="2", description="b", status="completed")]
    assert o._is_done() is True


def test_is_done_any_failed(tmp_path):
    o = _make_orchestrator(tmp_path)
    o.tasks = [TaskModel(id="1", description="a", status="completed"), TaskModel(id="2", description="b", status="failed")]
    assert o._is_done() is False


def test_is_done_pending(tmp_path):
    o = _make_orchestrator(tmp_path)
    o.tasks = [TaskModel(id="1", description="a", status="completed"), TaskModel(id="2", description="b", status="pending")]
    assert o._is_done() is False


def test_is_done_empty(tmp_path):
    o = _make_orchestrator(tmp_path)
    o.tasks = []
    assert o._is_done() is True


def test_get_tasks_returns_tracked_tasks(tmp_path):
    o = _make_orchestrator(tmp_path)
    tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    o.tasks = tasks
    assert o._get_tasks() == tasks


@pytest.mark.asyncio
async def test_max_cycles_halt(tmp_path, monkeypatch):
    monkeypatch.setattr("furrow.config.settings.max_cycles", 1)
    # also patch the imported settings in orchestrator module if referenced
    import furrow.core.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.settings, "max_cycles", 1)
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "ok"}')
    # planner uses client; mock planner
    o = _make_orchestrator(tmp_path, mock_client=mock_client)
    with patch.object(o.planner, "plan", new=AsyncMock(return_value=Plan(tasks=[], rationale="ok"))):
        await o.run()
    assert o.cycles == 1


@pytest.mark.asyncio
async def test_worker_parses_edits_and_writes_files():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({"edits": [{"path": "out.py", "content": "print('hi')"}], "summary": "wrote file"}))
    mock_client.write_file = AsyncMock()
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.list_files = MagicMock(return_value=[])
    task = TaskModel(id="1", description="do", files=[])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    mock_client.write_file.assert_awaited_with("out.py", "print('hi')")
    assert "wrote file" in result


@pytest.mark.asyncio
async def test_worker_handles_non_json_response():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="  plain text  ")
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.list_files = MagicMock(return_value=[])
    task = TaskModel(id="1", description="do", files=[])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    assert result == "plain text"


@pytest.mark.asyncio
async def test_ollama_provider_branch():
    with patch("furrow.llm.settings") as mock_settings:
        mock_settings.provider = Provider.OLLAMA
        mock_settings.model = "m"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.worker_model = "m"
        # build client with a custom settings object
        from furrow.config import Settings
        s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=s)
        resp = MagicMock()
        resp.json.return_value = {"message": {"content": "hi"}}
        mock_post = AsyncMock(return_value=resp)
        client._ollama = MagicMock()
        client._ollama.post = mock_post
        out = await client.complete("prompt", model="m")
        assert out == "hi"
        mock_post.assert_awaited()
        args, kwargs = mock_post.call_args
        assert args[0] == "/api/chat"
        assert kwargs["json"]["model"] == "m"


@pytest.mark.asyncio
async def test_state_save_and_load(tmp_path):
    sp = tmp_path / "state.json"
    mock_client = MagicMock()
    o = Orchestrator(goal="goal", client=mock_client, state_path=sp)
    o.tasks = [TaskModel(id="1", description="a", status="completed")]
    o.cycles = 3
    o._save_state()
    o2 = Orchestrator(goal="goal", client=mock_client, state_path=sp)
    assert o2.goal == "goal"
    assert o2.cycles == 3
    assert [t.id for t in o2.tasks] == ["1"]
