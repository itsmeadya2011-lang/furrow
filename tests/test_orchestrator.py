from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import (
    Provider,
    Settings,
    TaskModel,
    TestResult,
)
from furrow.core.orchestrator import Orchestrator
from furrow.agents.worker import WorkerAgent


class _FakeSettings:
    """Minimal stand-in for Settings used by fake clients."""

    def __init__(self) -> None:
        self.workspace = Path(".")
        self.worker_model = "fake-model"
        self.planner_model = "fake-model"
        self.tester_model = "fake-model"
        self.model = "fake-model"
        self.provider = Provider.ANTHROPIC
        self.max_cycles = 0


class _FakeClient:
    """Records written files and returns canned LLM responses."""

    def __init__(self, response: str) -> None:
        self.settings = _FakeSettings()
        self._response = response
        self.written: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return self._response

    async def read_file(self, path: str | Path) -> str:
        return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        self.written.append((str(path), content))

    def list_files(self, directory: str | Path) -> list[str]:
        return []


def test_get_tasks_returns_stored_tasks() -> None:
    o = Orchestrator(goal="x", client=_FakeClient("{}"))
    tasks = [TaskModel(id="1", description="do thing")]
    o.tasks = tasks
    assert o._get_tasks() is tasks
    assert o._get_tasks() == tasks


def test_done_requires_passed_tests() -> None:
    o = Orchestrator(goal="x", client=_FakeClient("{}"))
    o.tasks = [TaskModel(id="1", description="x", status="completed")]
    o.last_test_result = TestResult(passed=False, summary="nope", failures=["bad"])
    assert o._is_done() is False

    o.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is True


def test_done_false_when_no_tasks() -> None:
    o = Orchestrator(goal="x", client=_FakeClient("{}"))
    o.tasks = []
    o.last_test_result = None
    assert o._is_done() is False


def test_max_cycles_setting() -> None:
    s = Settings(max_cycles=3)
    o = Orchestrator(goal="x", client=_FakeClient("{}"), settings=s)
    assert o.settings.max_cycles == 3

    default = Settings()
    assert default.max_cycles == 0


def test_worker_parses_json_and_writes_files() -> None:
    payload = (
        '{"files": [{"path": "out/a.txt", "action": "create", "content": "hello"}],'
        ' "summary": "created a"}'
    )
    client = _FakeClient(payload)
    w = WorkerAgent(task=TaskModel(id="1", description="create a file"), client=client)
    import asyncio

    result = asyncio.run(w.run())
    assert "created a" in result
    assert len(client.written) == 1
    path, content = client.written[0]
    assert content == "hello"
    assert path.endswith("out/a.txt")


def test_worker_rejects_non_json() -> None:
    client = _FakeClient("not json at all")
    w = WorkerAgent(task=TaskModel(id="1", description="x"), client=client)
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(w.run())


def test_worker_rejects_missing_files_array() -> None:
    client = _FakeClient('{"summary": "no files"}')
    w = WorkerAgent(task=TaskModel(id="1", description="x"), client=client)
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(w.run())