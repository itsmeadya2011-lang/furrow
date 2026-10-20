"""Tests for the worker agent: the agent that actually edits files on disk."""

from __future__ import annotations

import json
from pathlib import Path

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from tests.helpers import FakeLLMClient, make_worker


def make_task(task_id: str = "1", files: list[str] | None = None) -> TaskModel:
    return TaskModel(id=task_id, description="write a file", files=files or [])


async def test_worker_writes_the_requested_file(settings: Settings, tmp_workspace: Path) -> None:
    payload = make_worker(summary="added a.txt", edits=[{"path": "a.txt", "content": "hello"}])
    client = FakeLLMClient(settings, workers=[payload])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    result = await worker.run()

    written = tmp_workspace / "a.txt"
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == "hello"
    assert worker.files_changed == ["a.txt"]
    assert "a.txt" in result


async def test_worker_reports_the_files_it_changed(settings: Settings, tmp_workspace: Path) -> None:
    payload = make_worker(
        edits=[
            {"path": "a.txt", "content": "one"},
            {"path": "nested/b.txt", "content": "two"},
        ]
    )
    client = FakeLLMClient(settings, workers=[payload])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    await worker.run()

    assert sorted(worker.files_changed) == ["a.txt", "nested/b.txt"]
    assert (tmp_workspace / "nested" / "b.txt").read_text(encoding="utf-8") == "two"


async def test_worker_rejects_a_path_that_escapes_the_workspace(
    settings: Settings, tmp_workspace: Path, tmp_path: Path
) -> None:
    payload = make_worker(edits=[{"path": "../escape.txt", "content": "nope"}])
    client = FakeLLMClient(settings, workers=[payload])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    await worker.run()

    assert not (tmp_path / "escape.txt").exists()
    assert worker.files_changed == []
    assert "../escape.txt" in worker.rejected_edits


async def test_worker_rejects_an_absolute_path_outside_the_workspace(
    settings: Settings, tmp_workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "absolute_escape.txt"
    payload = make_worker(edits=[{"path": str(outside), "content": "nope"}])
    client = FakeLLMClient(settings, workers=[payload])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    await worker.run()

    assert not outside.exists()
    assert worker.files_changed == []
    assert str(outside) in worker.rejected_edits


async def test_worker_ignores_a_prose_response(settings: Settings, tmp_workspace: Path) -> None:
    client = FakeLLMClient(settings, workers=["I could not finish the task, sorry."])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    result = await worker.run()

    assert list(tmp_workspace.iterdir()) == []
    assert worker.files_changed == []
    assert worker.rejected_edits == []
    assert "could not finish" in result


async def test_worker_writes_nothing_when_apply_worker_edits_is_false(
    settings: Settings, tmp_workspace: Path
) -> None:
    settings.apply_worker_edits = False
    payload = make_worker(edits=[{"path": "a.txt", "content": "hello"}])
    client = FakeLLMClient(settings, workers=[payload])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    await worker.run()

    assert not (tmp_workspace / "a.txt").exists()
    assert worker.files_changed == []
    assert "a.txt" in worker.rejected_edits


async def test_worker_skips_a_malformed_edit(settings: Settings, tmp_workspace: Path) -> None:
    payload = make_worker(edits=[{"path": "b.txt", "content": 42}])
    client = FakeLLMClient(settings, workers=[payload])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    await worker.run()

    assert not (tmp_workspace / "b.txt").exists()
    assert worker.files_changed == []
    assert worker.rejected_edits == ["b.txt"]


async def test_worker_skips_an_edit_without_a_path(settings: Settings, tmp_workspace: Path) -> None:
    payload = json.dumps({"summary": "s", "edits": [{"content": "orphan"}]})
    client = FakeLLMClient(settings, workers=[payload])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)

    await worker.run()

    assert worker.files_changed == []
    assert worker.rejected_edits == ["<missing>"]


async def test_worker_tolerates_a_failing_llm_call(settings: Settings, tmp_workspace: Path) -> None:
    client = FakeLLMClient(settings, workers=[])
    worker = WorkerAgent(task=make_task(), client=client, workspace=tmp_workspace)
    client.workers.clear()

    async def _boom(prompt: str, system: str = "", model: str | None = None) -> str:
        raise RuntimeError("provider exploded")

    client.response_fn = _boom

    result = await worker.run()

    assert "Worker failed" in result
    assert worker.files_changed == []


async def test_worker_includes_the_task_context_in_the_prompt(
    settings: Settings, tmp_workspace: Path
) -> None:
    (tmp_workspace / "existing.txt").write_text("existing content", encoding="utf-8")
    payload = make_worker(edits=[])
    client = FakeLLMClient(settings, workers=[payload])
    task = make_task(files=["existing.txt"])
    worker = WorkerAgent(task=task, client=client, workspace=tmp_workspace)

    await worker.run()

    prompt = client.calls[-1]["prompt"]
    assert "write a file" in prompt
    assert "existing content" in prompt
    assert str(tmp_workspace) in prompt
