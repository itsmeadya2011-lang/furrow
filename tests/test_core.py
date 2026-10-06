import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


async def test_complete_unsupported_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


def test_anthropic_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(settings=Settings(anthropic_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_openai_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)
    client = LLMClient(settings=Settings(openai_api_key=None))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


async def test_complete_anthropic():
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="anthropic reply")])
    )
    with patch("furrow.llm.AsyncAnthropic", return_value=mock_anthropic):
        client = LLMClient(
            settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="sk-test")
        )
        result = await client.complete("do something", system="be nice", model="claude-test")

    assert result == "anthropic reply"
    mock_anthropic.messages.create.assert_awaited_once()
    kwargs = mock_anthropic.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-test"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["system"] == "be nice"
    assert kwargs["messages"] == [{"role": "user", "content": "do something"}]


async def test_complete_openai():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai reply"))]
        )
    )
    with patch("furrow.llm.AsyncOpenAI", return_value=mock_openai):
        client = LLMClient(
            settings=Settings(provider=Provider.OPENAI, openai_api_key="sk-test")
        )
        result = await client.complete("do something", system="be nice", model="gpt-test")

    assert result == "openai reply"
    mock_openai.chat.completions.create.assert_awaited_once()
    kwargs = mock_openai.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "do something"},
    ]


async def test_read_write_file(tmp_path):
    client = LLMClient(settings=Settings())
    path = tmp_path / "nested" / "file.txt"

    await client.write_file(path, "hello world")

    assert path.read_text() == "hello world"
    assert await client.read_file(path) == "hello world"


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    client = LLMClient(settings=Settings())

    files = client.list_files(tmp_path)

    assert sorted(files) == ["a.txt", "sub/b.txt"]
    assert client.list_files(tmp_path / "missing") == []


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


async def test_plan_valid_json():
    mock_client = MagicMock()
    mock_client.settings = Settings(planner_model="planner-model")
    payload = {
        "tasks": [{"id": "1", "description": "do thing", "files": ["a.py"]}],
        "rationale": "ok",
    }
    mock_client.complete = AsyncMock(return_value=json.dumps(payload))
    planner = PlannerAgent(client=mock_client)

    plan = await planner.plan("build a feature")

    assert isinstance(plan, Plan)
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "ok"
    call = mock_client.complete.await_args
    assert "build a feature" in call.args[0]
    assert call.kwargs["model"] == "planner-model"


async def test_plan_invalid_json():
    mock_client = MagicMock()
    mock_client.settings = Settings()
    mock_client.complete = AsyncMock(return_value="this is not json")
    planner = PlannerAgent(client=mock_client)

    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("goal")


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------


async def test_worker_run_calls_complete(tmp_path):
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=tmp_path, worker_model="worker-model")
    mock_client.list_files.return_value = []
    mock_client.read_file = AsyncMock(side_effect=FileNotFoundError)
    mock_client.write_file = AsyncMock()
    mock_client.complete = AsyncMock(return_value="No edits produced.")

    task = TaskModel(id="1", description="implement feature", files=["src/main.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()

    mock_client.complete.assert_awaited_once()
    call = mock_client.complete.await_args
    assert "implement feature" in call.args[0]
    assert "src/main.py" in call.args[0]
    assert call.kwargs["model"] == "worker-model"
    assert "No file changes" in result


def test_parse_edits(tmp_path):
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=tmp_path)
    worker = WorkerAgent(task=TaskModel(id="1", description="x"), client=mock_client)

    response = (
        "FILE: src/foo.py\n"
        "```python\n"
        "print('hi')\n"
        "```\n"
        "END FILE\n"
        "FILE: /abs/bar.py\n"
        "```\n"
        "x = 1\n"
        "```\n"
        "END FILE\n"
    )
    edits = worker._parse_edits(response)

    assert len(edits) == 2
    rel_path, content = edits[0]
    assert rel_path == tmp_path / "src/foo.py"
    assert "print('hi')" in content
    assert edits[1][0] == Path("/abs/bar.py")
    assert edits[1][1] == "x = 1\n"


# ---------------------------------------------------------------------------
# TesterAgent
# ---------------------------------------------------------------------------


def _make_tester(tmp_path):
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=tmp_path)
    return TesterAgent(client=mock_client)


def test_detect_pytest(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\naddopts = '-q'\n")

    assert _make_tester(tmp_path)._detect_test_command(str(tmp_path)) == ["pytest", "-q"]


def test_detect_npm(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')

    assert _make_tester(tmp_path)._detect_test_command(str(tmp_path)) == [
        "npm",
        "test",
        "--",
        "--silent",
    ]


def test_detect_none(tmp_path):
    assert _make_tester(tmp_path)._detect_test_command(str(tmp_path)) is None


async def test_run_valid_json(tmp_path):
    mock_client = MagicMock()
    mock_client.settings = Settings(workspace=tmp_path)
    mock_client.complete = AsyncMock(
        return_value=json.dumps({"passed": True, "summary": "all passing", "failures": []})
    )
    tester = TesterAgent(client=mock_client)

    result = await tester.run("goal", [])

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "all passing"
    assert result.failures == []
    call = mock_client.complete.await_args
    assert "Goal: goal" in call.args[0]
    assert "No test runner found." in call.args[0]
    assert call.kwargs["model"] == mock_client.settings.tester_model


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_is_done(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch = Orchestrator(goal="g", client=MagicMock())

    orch.current_tasks = [TaskModel(id="1", description="a", status="completed")]
    assert orch._is_done() is True

    orch.current_tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is False

    orch.current_tasks = [TaskModel(id="1", description="a", status="failed")]
    assert orch._is_done() is False


def test_get_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch = Orchestrator(goal="g", client=MagicMock())
    tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]

    orch.current_tasks = tasks

    assert orch._get_tasks() == tasks


async def test_max_cycles_enforcement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    client.settings = Settings(max_cycles=2)
    orch = Orchestrator(goal="g", client=client)
    orch._cycle = AsyncMock()
    orch._is_done = lambda: False

    await orch.run()

    assert orch.cycles == 2
    assert orch._cycle.await_count == 2


def test_save_and_load_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    orch = Orchestrator(goal="original goal", client=client)
    orch.cycles = 3
    orch.current_tasks = [
        TaskModel(id="1", description="a", status="completed", result="done"),
    ]

    orch._save_state()

    state_file = tmp_path / ".furrow" / "state.json"
    assert state_file.exists()
    loaded = json.loads(state_file.read_text())
    assert loaded["goal"] == "original goal"
    assert loaded["cycles"] == 3
    assert loaded["current_tasks"][0]["id"] == "1"

    orch2 = Orchestrator(goal="new goal", client=client)
    assert orch2.goal == "original goal"
    assert orch2.cycles == 3
    assert orch2.current_tasks[0]["id"] == "1"
    assert orch2.current_tasks[0]["status"] == "completed"
