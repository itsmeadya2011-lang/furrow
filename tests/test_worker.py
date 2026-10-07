import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class MockLLMClient:
    def __init__(self):
        self.settings = type("Settings", (), {"worker_model": "test-model"})()

    async def complete(self, prompt: str, model: str | None = None) -> str:
        return "Done"

    async def read_file(self, path: str) -> str:
        return "original content"

    async def write_file(self, path: str, content: str) -> None:
        pass


@pytest.mark.asyncio
async def test_worker_initialization():
    task = TaskModel(id="1", description="test task")
    client = MockLLMClient()
    worker = WorkerAgent(task=task, client=client)
    assert worker.task.id == "1"
    assert worker.max_rounds == 3


@pytest.mark.asyncio
async def test_worker_parse_edits():
    task = TaskModel(id="1", description="test task")
    client = MockLLMClient()
    worker = WorkerAgent(task=task, client=client)

    response = """```edit
path: test.py
<<<<<<< ORIGINAL
old code
=======
new code
>>>>>>> UPDATED
```"""

    edits = worker._parse_edits(response)
    assert len(edits) == 1
    assert edits[0]["path"] == "test.py"
    assert edits[0]["original"] == "old code"
    assert edits[0]["updated"] == "new code"


@pytest.mark.asyncio
async def test_worker_parse_multiple_edits():
    task = TaskModel(id="1", description="test task")
    client = MockLLMClient()
    worker = WorkerAgent(task=task, client=client)

    response = """```edit
path: a.py
<<<<<<< ORIGINAL
old a
=======
new a
>>>>>>> UPDATED
```
```edit
path: b.py
<<<<<<< ORIGINAL
old b
=======
new b
>>>>>>> UPDATED
```"""

    edits = worker._parse_edits(response)
    assert len(edits) == 2
    assert edits[0]["path"] == "a.py"
    assert edits[1]["path"] == "b.py"


@pytest.mark.asyncio
async def test_worker_parse_no_edits():
    task = TaskModel(id="1", description="test task")
    client = MockLLMClient()
    worker = WorkerAgent(task=task, client=client)

    response = "Just a summary, no edits"
    edits = worker._parse_edits(response)
    assert len(edits) == 0


@pytest.mark.asyncio
async def test_worker_parse_malformed_edit():
    task = TaskModel(id="1", description="test task")
    client = MockLLMClient()
    worker = WorkerAgent(task=task, client=client)

    response = """```edit
path: test.py
<<<<<<< ORIGINAL
old code
=======
new code
```"""  # Missing >>>>>>> UPDATED

    edits = worker._parse_edits(response)
    assert len(edits) == 0