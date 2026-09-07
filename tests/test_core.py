import json
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, patch

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_init():
    orch = Orchestrator(goal="test goal")
    assert orch.goal == "test goal"
    assert orch.cycles == 0
    assert orch.tasks == []


def test_orchestrator_is_done_no_tasks():
    orch = Orchestrator(goal="test")
    assert not orch._is_done()


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done()


def test_orchestrator_is_done_with_failures():
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert not orch._is_done()


def test_orchestrator_save_state(tmp_path: Path):
    orch = Orchestrator(goal="test")
    orch.state_file = tmp_path / "state.json"
    orch.tasks = [TaskModel(id="1", description="do it", status="completed", result="done")]
    orch._save_state()
    data = json.loads((tmp_path / "state.json").read_text())
    assert data["goal"] == "test"
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["status"] == "completed"


def test_orchestrator_load_state(tmp_path: Path):
    state = {"goal": "loaded goal", "cycles": 3, "tasks": []}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state))
    orch = Orchestrator(goal="old")
    orch.state_file = p
    loaded = orch.load_state()
    assert loaded
    assert orch.goal == "loaded goal"
    assert orch.cycles == 3


@pytest.mark.asyncio
async def test_worker_returns_dict():
    client = AsyncMock()
    client.settings = Settings(worker_model="test-model")
    client.complete = AsyncMock(return_value='{"files": {"test.py": "print(1)"}, "summary": "done"}')
    client.write_file = AsyncMock()
    task = TaskModel(id="1", description="write test.py")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert "test.py" in result["files_written"]


@pytest.mark.asyncio
async def test_worker_fallback_on_bad_json():
    client = AsyncMock()
    client.settings = Settings(worker_model="test-model")
    client.complete = AsyncMock(return_value="not json")
    task = TaskModel(id="1", description="do stuff")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert result["files_written"] == []


@pytest.mark.asyncio
async def test_tester_no_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    client = AsyncMock()
    client.settings = Settings(tester_model="test-model")
    client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
    tester = TesterAgent(client=client)
    result = await tester.run("goal", [])
    assert isinstance(result, TestResult)
    assert result.passed


@pytest.mark.asyncio
async def test_llm_client_ollama():
    import httpx
    client = LLMClient(settings=Settings(provider="ollama", model="llama3", ollama_base_url="http://localhost:11434"))
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value.json.return_value = {"response": "hello"}
        result = await client._complete_ollama("hi", "", "llama3")
        assert result == "hello"
