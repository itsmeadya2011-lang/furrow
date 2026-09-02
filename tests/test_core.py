"""Tests for Furrow core: config models, LLM client selection, WorkerAgent, Orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncMock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import (
    FileOperation,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    WorkerResult,
    settings,
)
from furrow.core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# config / models
# ---------------------------------------------------------------------------
def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_file_operation_write():
    op = FileOperation(path="foo.py", content="print('hi')")
    assert op.is_edit is False


def test_file_operation_edit():
    op = FileOperation(path="foo.py", old_str="a", new_str="b")
    assert op.is_edit is True


def test_file_operation_mutual_exclusion():
    with pytest.raises(ValueError, match="cannot specify both"):
        FileOperation(path="x", content="y", old_str="a", new_str="b")


def test_file_operation_neither_provided():
    with pytest.raises(ValueError, match="provide content"):
        FileOperation(path="x")


def test_worker_result_defaults():
    r = WorkerResult(summary="done")
    assert r.operations == []
    assert r.issues == []


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_worker_agent_writes_new_file(tmp_path: Path):
    """WorkerAgent should parse LLM JSON and write new files to disk."""
    task = TaskModel(id="t1", description="create file", files=["foo.py"])
    mock_client = MagicMock()
    mock_client.settings = settings
    mock_client.read_file = AsyncMock(
        side_effect=FileNotFoundError("no prior file")
    )
    mock_client.write_file = AsyncMock()
    mock_client.complete = AsyncMock(
        return_value="""```json
{"summary": "created foo.py", "operations": [{"path": "foo.py", "content": "x = 1\\n"}], "issues": []}
```"""
    )

    from furrow.agents.worker import WorkerAgent

    worker = WorkerAgent(task=task, client=mock_client, workspace=tmp_path)
    summary = await worker.run()

    mock_client.write_file.assert_awaited_once()
    assert "created foo.py" in summary
    written_path, written_content = mock_client.write_file.call_args.args
    assert written_content == "x = 1\n"


@pytest.mark.asyncio
async def test_worker_agent_edits_existing_file(tmp_path: Path):
    """WorkerAgent should patch existing files when content already exists."""
    existing = tmp_path / "bar.py"
    existing.write_text("old line\n")
    task = TaskModel(id="t2", description="edit", files=["bar.py"])
    mock_client = MagicMock()
    mock_client.settings = settings
    mock_client.read_file = AsyncMock(return_value="old line\n")
    mock_client.write_file = AsyncMock()
    mock_client.complete = AsyncMock(
        return_value=(
            '{"summary":"patched","operations":[{"path":"bar.py",'
            '"old_str":"old line","new_str":"new line"}],"issues":[]}'
        )
    )

    from furrow.agents.worker import WorkerAgent

    worker = WorkerAgent(task=task, client=mock_client, workspace=tmp_path)
    await worker.run()

    mock_client.write_file.assert_awaited_once()
    _, content = mock_client.write_file.call_args.args
    assert content == "new line\n"


@pytest.mark.asyncio
async def test_worker_agent_invalid_json(tmp_path: Path):
    """Invalid JSON should produce a safe result with an issue recorded."""
    task = TaskModel(id="t3", description="bad", files=[])
    mock_client = MagicMock()
    mock_client.settings = settings
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.write_file = AsyncMock()
    mock_client.complete = AsyncMock(return_value="not json at all {{{")

    from furrow.agents.worker import WorkerAgent

    worker = WorkerAgent(task=task, client=mock_client, workspace=tmp_path)
    summary = await worker.run()
    assert "Failed" in summary
    mock_client.write_file.assert_not_awaited()


# ---------------------------------------------------------------------------
# LLMClient provider selection
# ---------------------------------------------------------------------------
def _make_settings(provider: Provider, **overrides) -> Settings:
    values = {"provider": provider, "model": "test-model", "ollama_base_url": "http://localhost:11434"}
    values.update(overrides)
    return Settings(**values)


def test_llm_client_ollama_supported():
    s = _make_settings(Provider.OLLAMA, ollama_base_url="http://test:9999")
    from furrow.llm import LLMClient

    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OLLAMA


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider_raises():
    s = Settings(provider="bogus")  # type: ignore[arg-type]  # will be coerced...
    # Provider("bogus") raises ValueError, so instead test by patching
    s = Settings()
    s.provider = Provider("bogus") if False else Provider.ANTHROPIC
    # Direct approach: set an invalid provider via object.__setattr__
    object.__setattr__(s, "provider", "bogus")
    from furrow.llm import LLMClient

    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test")
    orch._all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_not_done_when_pending():
    orch = Orchestrator(goal="test")
    orch._all_tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_not_done_when_failed():
    orch = Orchestrator(goal="test")
    orch._all_tasks = [TaskModel(id="1", description="a", status="failed", result="err")]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_empty_no():
    orch = Orchestrator(goal="test")
    # No tasks yet -> not done
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_get_tasks_stores_across_cycles():
    orch = Orchestrator(goal="test")
    t_before = orch._get_tasks()
    assert t_before == []
    orch._all_tasks = [TaskModel(id="1", description="a")]
    assert len(orch._get_tasks()) == 1


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_enforcement(tmp_path: Path):
    """When max_cycles is set, the orchestrator should stop after N cycles."""
    s = Settings(
        provider=Provider.ANTHROPIC,
        max_cycles=1,
        max_parallel_tasks=2,
        workspace=tmp_path,
    )
    mock_client = MagicMock()
    mock_client.settings = s
    mock_client.complete = AsyncMock()
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.write_file = AsyncMock()

    # Plan returns no tasks so cycle completes immediately; with max_cycles=1
    # the loop should run exactly 1 cycle then stop.
    plan = Plan(tasks=[], rationale="none")
    mock_client.complete.return_value = '{"tasks": [], "rationale": "none"}'

    outputs: list[str] = []

    async def cb(text: str) -> None:
        outputs.append(text)

    orch = Orchestrator(
        goal="noop", client=mock_client, settings=s, on_output=cb
    )
    await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_zero_runs_forever_until_done(tmp_path: Path):
    """max_cycles=0 means unlimited; loop exits when _is_done() is True."""
    s = Settings(
        provider=Provider.ANTHROPIC,
        max_cycles=0,
        max_parallel_tasks=1,
        workspace=tmp_path,
    )
    mock_client = MagicMock()
    mock_client.settings = s
    # First cycle returns a task, second cycle returns empty plan -> done
    mock_client.complete = AsyncMock(
        side_effect=[
            '{"tasks": [{"id": "1", "description": "do", "files": [], "dependencies": []}],"rationale":"x"}',
            '{"tasks": [], "rationale": "complete"}',
        ]
    )
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.write_file = AsyncMock()

    orch = Orchestrator(goal="done", client=mock_client, settings=s)

    async with patch.object(orch, "_cycle", wraps=orch._cycle) as mock_cycle:
        await orch.run()

    # 2 cycles: first plans a task, second finds no tasks -> _is_done True
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_parallel_semaphore_uses_setting():
    orch = Orchestrator(goal="test")
    assert orch._semaphore is None
    await orch.start()
    assert orch._semaphore is not None
    # Default max_parallel_tasks is 5
    assert orch._semaphore._value == 5


@pytest.mark.asyncio
async def test_orchestrator_start_sets_workspace(tmp_path: Path):
    orch = Orchestrator(goal="test")
    await orch.start(workspace=tmp_path)
    assert orch.settings.workspace == tmp_path
