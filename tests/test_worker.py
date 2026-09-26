from __future__ import annotations

import json

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


async def test_worker_writes_files(mock_client) -> None:
    mock_client.complete.return_value = json.dumps(
        {
            "summary": "wrote files",
            "files": {"a.py": "content-a", "b.py": "content-b"},
        }
    )
    task = TaskModel(id="1", description="write stuff", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task, client=mock_client)

    result = await worker.run()

    assert "wrote files" in result
    mock_client.write_file.assert_any_await("a.py", "content-a")
    mock_client.write_file.assert_any_await("b.py", "content-b")
    assert mock_client.write_file.await_count == 2


async def test_worker_no_files_in_response(mock_client) -> None:
    mock_client.complete.return_value = json.dumps({"files": {}, "summary": "nothing to do"})
    task = TaskModel(id="1", description="no-op")
    worker = WorkerAgent(task=task, client=mock_client)

    result = await worker.run()

    mock_client.write_file.assert_not_awaited()
    assert "nothing to do" in result
    assert "none" in result


async def test_worker_handles_non_json(mock_client) -> None:
    mock_client.complete.return_value = "this is not valid json <<<"
    task = TaskModel(id="1", description="t")
    worker = WorkerAgent(task=task, client=mock_client)

    result = await worker.run()

    assert "Error" in result
    assert "Failed to parse LLM response as JSON" in result
    assert "this is not valid json <<<" in result


async def test_worker_reads_context_files(mock_client) -> None:
    mock_client.read_file.return_value = "previous content"
    mock_client.complete.return_value = json.dumps({"files": {}, "summary": "done"})
    task = TaskModel(id="1", description="t", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task, client=mock_client)

    await worker.run()

    assert mock_client.read_file.await_count == 2
    mock_client.read_file.assert_any_await("a.py")
    mock_client.read_file.assert_any_await("b.py")


async def test_worker_context_for_new_file(mock_client) -> None:
    mock_client.read_file.side_effect = FileNotFoundError
    mock_client.complete.return_value = json.dumps({"files": {}, "summary": "ok"})
    task = TaskModel(id="1", description="t", files=["new.py"])
    worker = WorkerAgent(task=task, client=mock_client)

    await worker.run()

    mock_client.read_file.assert_awaited_once_with("new.py")
    prompt = mock_client.complete.call_args[0][0]
    assert "(new file)" in prompt
