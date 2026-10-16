import asyncio
from pathlib import Path

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


class FakeLLMClient:
    """Stand-in for LLMClient that returns scripted responses and records I/O."""

    def __init__(self, responses=None, files=None) -> None:
        self.responses = list(responses or [])
        self.calls: list[str] = []
        self.files: dict[str, str] = dict(files or {})
        self.writes: list[tuple[str, str]] = []
        self.reads: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "FINISH"

    async def read_file(self, path: str | Path) -> str:
        key = str(path)
        self.reads.append(key)
        if key in self.files:
            return self.files[key]
        raise FileNotFoundError(key)

    async def write_file(self, path: str | Path, content: str) -> None:
        self.writes.append((str(path), content))

    def list_files(self, directory: str | Path) -> list[str]:
        return []


def _make_task(task_id: str, description: str = "do thing", files=None) -> TaskModel:
    return TaskModel(id=task_id, description=description, files=files or [])


@pytest.mark.asyncio
async def test_worker_writes_files_from_directives():
    client = FakeLLMClient(
        responses=[
            "WRITE: src/hello.py\n<<<CONTENT\nprint('hi')\n>>>CONTENT\nFINISH",
        ],
        files={"src/hello.py": "print('old')"},
    )
    settings = Settings(workspace=Path("/tmp/furrow-ws"))
    worker = WorkerAgent(task=_make_task("1", files=["src/hello.py"]), client=client, settings=settings)
    result = await worker.run()

    assert ("/tmp/furrow-ws/src/hello.py", "print('hi')") in client.writes
    assert "wrote 1 file(s)" in result
    assert "read 1 file(s)" in result


@pytest.mark.asyncio
async def test_worker_reads_then_writes():
    client = FakeLLMClient(
        responses=[
            "READ: src/hello.py",
            "WRITE: src/hello.py\n<<<CONTENT\nprint('updated')\n>>>CONTENT\nFINISH",
        ],
        files={"/tmp/furrow-ws/src/hello.py": "print('old')"},
    )
    settings = Settings(workspace=Path("/tmp/furrow-ws"))
    worker = WorkerAgent(task=_make_task("1"), client=client, settings=settings)
    await worker.run()

    assert len(client.writes) == 1
    assert client.writes[0][1] == "print('updated')"


@pytest.mark.asyncio
async def test_worker_finish_without_directives():
    client = FakeLLMClient(responses=["FINISH"])
    worker = WorkerAgent(task=_make_task("1"), client=client)
    result = await worker.run()
    assert "no changes made" in result
    assert client.writes == []


@pytest.mark.asyncio
async def test_worker_bounded_by_max_turns():
    # A response that never emits FINISH should stop after max_turns.
    client = FakeLLMClient(responses=["READ: src/a.py"] * 10)
    settings = Settings(max_turns=2, workspace=Path("/tmp/furrow-ws"))
    worker = WorkerAgent(task=_make_task("1"), client=client, settings=settings)
    await worker.run()
    assert len(client.reads) == 2


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_failed():
    t = TestResult(passed=False, summary="nope", failures=["a", "b"])
    assert t.passed is False
    assert t.failures == ["a", "b"]
