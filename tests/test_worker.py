"""Tests for the WorkerAgent file change parsing."""

from __future__ import annotations

import pytest
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class _FakeSettings(Settings):
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


class _FakeClient:
    """Minimal fake LLMClient for testing _apply_changes."""

    def __init__(self) -> None:
        self.written: list[tuple[str, str]] = []
        self.settings = _FakeSettings()

    async def write_file(self, path: str, content: str) -> None:
        self.written.append((path, content))

    async def complete(self, prompt: str, model: str | None = None) -> str:
        return ""


def test_apply_changes_parses_fenced_blocks():
    client = _FakeClient()
    agent = WorkerAgent(
        task=TaskModel(id="1", description="create file"),
        client=client,
    )
    response = """Here's the implementation:

```python
# src/main.py
def hello():
    return "world"
```

Done!
"""
    import asyncio

    asyncio.run(agent._apply_changes(response))
    assert len(client.written) == 1
    path, content = client.written[0]
    assert path == "src/main.py"
    assert "def hello():" in content


def test_apply_changes_multiple_files():
    client = _FakeClient()
    agent = WorkerAgent(
        task=TaskModel(id="1", description="create files"),
        client=client,
    )
    response = """```python
# src/a.py
x = 1
```

```python
# src/b.py
y = 2
```
"""
    import asyncio

    asyncio.run(agent._apply_changes(response))
    assert len(client.written) == 2
    paths = {p for p, _ in client.written}
    assert paths == {"src/a.py", "src/b.py"}


def test_apply_changes_no_code_blocks():
    client = _FakeClient()
    agent = WorkerAgent(
        task=TaskModel(id="1", description="do thing"),
        client=client,
    )
    response = "I don't have any code changes to make."
    import asyncio

    asyncio.run(agent._apply_changes(response))
    assert len(client.written) == 0


def test_apply_changes_handles_write_error():
    class _BadClient(_FakeClient):
        async def write_file(self, path: str, content: str) -> None:
            raise OSError("disk full")

    client = _BadClient()
    agent = WorkerAgent(
        task=TaskModel(id="1", description="do thing"),
        client=client,
    )
    response = """```python
# src/a.py
x = 1
```
"""
    import asyncio

    # Should not raise - best effort
    asyncio.run(agent._apply_changes(response))