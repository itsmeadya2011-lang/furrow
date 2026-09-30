from __future__ import annotations

from unittest import mock

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def test_parse_files_single_block():
    resp = "===FURROW_FILE: src/a.py===\nprint('hello')\n===FURROW_END==="
    assert WorkerAgent.parse_files(resp) == [("src/a.py", "print('hello')")]


def test_parse_files_multiple_blocks():
    resp = (
        "===FURROW_FILE: a.py===\n1\n===FURROW_END===\n"
        "===FURROW_FILE: b.py===\n2\n===FURROW_END==="
    )
    assert WorkerAgent.parse_files(resp) == [("a.py", "1"), ("b.py", "2")]


def test_parse_files_content_with_special_markers():
    content = "line: colon\n=== inside\n--- also dash\nmixed: x == y"
    resp = f"===FURROW_FILE: src/c.py===\n{content}\n===FURROW_END==="
    assert WorkerAgent.parse_files(resp) == [("src/c.py", content)]


def test_parse_files_no_blocks_returns_empty():
    assert WorkerAgent.parse_files("just some text, no sentinel blocks here") == []


def test_parse_files_path_surrounded_by_spaces():
    resp = "===FURROW_FILE:   src/d.py   ===\nhi\n===FURROW_END==="
    assert WorkerAgent.parse_files(resp) == [("src/d.py", "hi")]


async def test_run_writes_parsed_files():
    response = (
        "Here is the change.\n\n"
        "===FURROW_FILE: src/a.py===\nprint('hello')\n===FURROW_END===\n\n"
        "===FURROW_FILE: src/b.py===\nx = 1\n===FURROW_END===\n"
    )
    client = mock.MagicMock()
    client.settings = mock.Mock(worker_model="test-model")
    client.complete = mock.AsyncMock(return_value=response)
    client.write_file = mock.AsyncMock()

    task = TaskModel(id="1", description="add files")
    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()

    assert "Wrote 2 file(s)" in summary
    assert "src/a.py" in summary
    assert "src/b.py" in summary
    assert client.write_file.await_count == 2
    calls = client.write_file.call_args_list
    assert calls[0].args == ("src/a.py", "print('hello')")
    assert calls[1].args == ("src/b.py", "x = 1")
    client.complete.assert_awaited_once()


async def test_run_no_blocks_reports_no_changes():
    response = "I couldn't figure out any file changes."
    client = mock.MagicMock()
    client.settings = mock.Mock(worker_model="test-model")
    client.complete = mock.AsyncMock(return_value=response)
    client.write_file = mock.AsyncMock()

    task = TaskModel(id="2", description="nothing here")
    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()

    assert "No file changes were produced." in summary
    assert client.write_file.await_count == 0
