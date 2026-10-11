from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent, _parse_file_changes
from furrow.config import TaskModel


def test_parse_file_changes_single_file_python():
    response = """Here is the change:

FILE: src/main.py
```python
print("hello")
```

Done."""
    result = _parse_file_changes(response)
    assert result == {"src/main.py": 'print("hello")'}


def test_parse_file_changes_multiple_files():
    response = """FILE: a.py
```python
x = 1
```

FILE: b.py
```python
y = 2
```"""
    result = _parse_file_changes(response)
    assert result == {"a.py": "x = 1", "b.py": "y = 2"}


def test_parse_file_changes_different_language_tags():
    response = """FILE: script.js
```javascript
console.log("hi");
```

FILE: notes.md
```markdown
# Title
```"""
    result = _parse_file_changes(response)
    assert result == {"script.js": 'console.log("hi");', "notes.md": "# Title"}


def test_parse_file_changes_no_file_blocks():
    response = "Just some text without any file blocks."
    result = _parse_file_changes(response)
    assert result == {}


async def test_read_context_returns_file_contents(tmp_path):
    file_a = tmp_path / "a.py"
    file_a.write_text("print('a')")
    file_b = tmp_path / "b.py"
    file_b.write_text("print('b')")

    mock_client = MagicMock()

    async def _fake_read(path):
        with open(path) as f:
            return f.read()

    mock_client.read_file = AsyncMock(side_effect=_fake_read)

    task = TaskModel(id="1", description="test", files=[str(file_a), str(file_b)])
    worker = WorkerAgent(task=task, client=mock_client)
    context = await worker._read_context()
    assert "print('a')" in context
    assert "print('b')" in context


async def test_read_context_empty_when_no_files():
    task = TaskModel(id="1", description="test", files=[])
    mock_client = MagicMock()
    worker = WorkerAgent(task=task, client=mock_client)
    context = await worker._read_context()
    assert context == ""