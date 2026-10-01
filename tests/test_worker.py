"""Tests for WorkerAgent file parsing and writing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import _FILE_BLOCK_RE, WorkerAgent
from furrow.config import TaskModel


def _make_client(tmp_path: Path) -> MagicMock:
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "test-worker"
    client.settings.goal = ""
    client.settings.workspace = tmp_path
    client.write_file = AsyncMock()
    client.resolve_path = MagicMock(side_effect=lambda p: _resolve(p, tmp_path))
    client.complete = AsyncMock(return_value="")
    return client


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


class TestFileBlockRegex:
    def test_single_file_block(self):
        text = """Here's the code:
{FILE: src/main.py}
```python
print("hello")
```
{DONE}"""
        matches = _FILE_BLOCK_RE.findall(text)
        assert len(matches) == 1
        # Should have path, content groups
        match = _FILE_BLOCK_RE.search(text)
        assert match.group("path") == "src/main.py"
        assert 'print("hello")' in match.group("content")

    def test_multiple_file_blocks(self):
        text = """{FILE: src/a.py}
```python
content_a = 1
```
{FILE: src/b.py}
```python
content_b = 2
```"""
        matches = _FILE_BLOCK_RE.findall(text)
        assert len(matches) == 2

    def test_no_file_block_returns_empty(self):
        text = "Just a regular message with no files."
        assert _FILE_BLOCK_RE.findall(text) == []

    def test_file_block_with_language(self):
        text = """{FILE: app.js}
```javascript
const x = 1;
```"""
        match = _FILE_BLOCK_RE.search(text)
        assert match is not None
        assert match.group("path") == "app.js"
        assert "const x" in match.group("content")


class TestWorkerRun:
    @pytest.mark.asyncio
    async def test_run_writes_files(self, tmp_path):
        client = _make_client(tmp_path)
        client.complete.return_value = """Implementation done.

{FILE: src/main.py}
```python
def hello():
    return "world"
```"""
        task = TaskModel(id="1", description="Create hello function", files=["src/main.py"])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()

        assert "src/main.py" in result
        assert client.write_file.called

    @pytest.mark.asyncio
    async def test_run_no_files(self, tmp_path):
        client = _make_client(tmp_path)
        client.complete.return_value = "Task completed successfully. No files needed."
        task = TaskModel(id="2", description="Research task")
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()

        assert "Files written" not in result
        assert "Task completed successfully" in result

    @pytest.mark.asyncio
    async def test_run_strips_file_blocks_from_summary(self, tmp_path):
        client = _make_client(tmp_path)
        client.complete.return_value = """{FILE: src/x.py}
```python
x = 1
```
Done."""
        task = TaskModel(id="3", description="write file")
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()

        # FILE blocks should not appear in summary
        assert "{FILE:" not in result
        assert "src/x.py" in result  # but path is listed as written
        assert "Done." in result
