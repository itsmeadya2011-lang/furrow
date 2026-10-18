import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.mark.asyncio
async def test_apply_changes_parses(tmp_path):
    client = MagicMock()
    client.write_file = AsyncMock()
    task = TaskModel(id="1", description="d")
    worker = WorkerAgent(task=task, client=client)
    resp = "```filename: a.txt\nhello\n```"
    changed = await worker._apply_changes(resp)
    assert changed == ["a.txt"]
    client.write_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_files_written_correctly(tmp_path):
    client = MagicMock()
    client.write_file = AsyncMock()
    task = TaskModel(id="1", description="d")
    worker = WorkerAgent(task=task, client=client)
    resp = "```filename: b.txt\ncontent\n```"
    await worker._apply_changes(resp)
    client.write_file.assert_awaited_with("b.txt", "content\n")


@pytest.mark.asyncio
async def test_nonexistent_files_handled():
    client = MagicMock()
    client.list_files = MagicMock(return_value=[])
    client.read_file = AsyncMock(side_effect=FileNotFoundError)
    client.write_file = AsyncMock()
    client.complete = AsyncMock(return_value="no changes")
    client.settings.workspace = "/tmp/doesnotexist"
    task = TaskModel(id="1", description="d", files=["missing.txt"])
    worker = WorkerAgent(task=task, client=client)
    res = await worker.run()
    assert "No file changes" in res
