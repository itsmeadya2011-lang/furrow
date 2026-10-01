from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def test_parse_file_edits_single():
    worker = WorkerAgent(task=TaskModel(id="1", description="do something"))
    response = '<FILE path="src/main.py">\nprint("hello")\n</FILE>'
    edits = worker._parse_file_edits(response)
    assert len(edits) == 1
    assert edits[0][0] == "src/main.py"
    assert edits[0][1] == 'print("hello")'


def test_parse_file_edits_multiple():
    worker = WorkerAgent(task=TaskModel(id="1", description="do something"))
    response = '<FILE path="a.py">\nprint(1)\n</FILE>\n<FILE path="b.py">\nprint(2)\n</FILE>'
    edits = worker._parse_file_edits(response)
    assert len(edits) == 2
    assert edits[0] == ("a.py", "print(1)")
    assert edits[1] == ("b.py", "print(2)")


def test_parse_file_edits_with_code_fences():
    worker = WorkerAgent(task=TaskModel(id="1", description="do something"))
    response = '<FILE path="main.py">\n```python\nprint(1)\n```\n</FILE>'
    edits = worker._parse_file_edits(response)
    assert len(edits) == 1
    assert edits[0][0] == "main.py"
    assert edits[0][1] == "print(1)"


def test_parse_file_edits_none():
    worker = WorkerAgent(task=TaskModel(id="1", description="do something"))
    response = "No file blocks here, just a summary of changes."
    edits = worker._parse_file_edits(response)
    assert edits == []


def test_parse_file_edits_with_indentation():
    worker = WorkerAgent(task=TaskModel(id="1", description="do something"))
    response = '<FILE path="main.py">\n    print(1)\n    print(2)\n</FILE>'
    edits = worker._parse_file_edits(response)
    assert len(edits) == 1
    assert edits[0][0] == "main.py"
    assert edits[0][1] == "print(1)\nprint(2)"


async def test_write_file_edits():
    client = MagicMock()
    client.write_file = AsyncMock()
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    response = '<FILE path="src/main.py">\nprint(1)\n</FILE>'
    results = await worker._write_file_edits(response)
    assert len(results) == 1
    assert results[0][0] == "src/main.py"
    assert results[0][1] is True
    client.write_file.assert_awaited_once_with("src/main.py", "print(1)")


async def test_worker_run_writes_files():
    client = MagicMock()
    client.read_file = AsyncMock()
    client.write_file = AsyncMock()
    client.complete = AsyncMock(return_value='<FILE path="src/main.py">\nprint(1)\n</FILE>')
    task = TaskModel(id="1", description="do something", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    client.write_file.assert_awaited_once_with("src/main.py", "print(1)")
    assert "Edited 1 file(s)" in result
    assert "src/main.py" in result
