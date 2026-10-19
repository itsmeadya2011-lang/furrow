import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.writes = []
        self.settings = Settings(workspace=__import__("pathlib").Path("/tmp"))

    async def complete(self, prompt, system="", model=None):
        return self.response

    async def read_file(self, path):
        return ""

    async def write_file(self, path, content):
        self.writes.append((str(path), content))

    def list_files(self, directory):
        return []


@pytest.mark.asyncio
async def test_worker_parses_file_blocks():
    client = FakeClient("FILE: foo.py\n```python\nprint('hi')\n```")
    task = TaskModel(id="1", description="t", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert client.writes
    assert client.writes[0][1].strip() == "print('hi')"
    assert "Changed files" in result


@pytest.mark.asyncio
async def test_worker_no_files_returns_raw():
    text = "plain response"
    client = FakeClient(text)
    task = TaskModel(id="1", description="t", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == text
    assert client.writes == []