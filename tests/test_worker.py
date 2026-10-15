from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def _make_worker(task_files=None, read_contents=None, llm_response=""):
    task = TaskModel(id="1", description="test task", files=task_files or [])
    client = MagicMock()
    client.settings.worker_model = "test-model"
    client.read_file = AsyncMock(side_effect=lambda p: (read_contents or {}).get(str(p), ""))
    client.write_file = AsyncMock()
    client.complete = AsyncMock(return_value=llm_response)
    return WorkerAgent(task=task, client=client, workspace=Path("/tmp")), client


def test_worker_reads_task_files():
    worker, client = _make_worker(
        task_files=["a.py", "b.py"],
        read_contents={"/tmp/a.py": "print('a')", "/tmp/b.py": "print('b')"},
    )
    import asyncio
    contents = asyncio.run(worker._read_task_files())
    assert contents["a.py"] == "print('a')"
    assert contents["b.py"] == "print('b')"


def test_worker_missing_file_treated_as_empty():
    worker, _ = _make_worker(task_files=["new.py"], read_contents={})
    import asyncio
    contents = asyncio.run(worker._read_task_files())
    assert contents["new.py"] == ""


def test_worker_parse_changes_extracts_file_blocks():
    worker, _ = _make_worker()
    response = '''Some preamble.

FILE: src/app.py
```python
print("hello")
```

FILE: tests/test_app.py
```python
def test_app():
    assert True
```
'''
    changes = worker._parse_changes(response)
    assert len(changes) == 2
    assert changes[0][0] == "src/app.py"
    assert 'print("hello")' in changes[0][1]
    assert changes[1][0] == "tests/test_app.py"


def test_worker_no_changes_returns_message():
    worker, _ = _make_worker(llm_response="I couldn't make any changes.")
    import asyncio
    result = asyncio.run(worker.run())
    assert "No file changes" in result


def test_worker_writes_parsed_changes():
    worker, client = _make_worker(
        task_files=["a.py"],
        read_contents={"/tmp/a.py": "old"},
        llm_response='FILE: a.py\n```python\nnew content\n```',
    )
    import asyncio
    result = asyncio.run(worker.run())
    client.write_file.assert_called_once()
    args = client.write_file.call_args
    assert args[0][0] == Path("/tmp/a.py")
    assert args[0][1] == "new content"
    assert "a.py" in result