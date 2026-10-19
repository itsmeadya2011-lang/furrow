import asyncio
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.worker import _FILE_BLOCK_RE, WorkerAgent
from furrow.config import Settings, TaskModel


def make_worker(response, files=None):
    s = Settings(_env_file=None)
    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    client.read_file = AsyncMock(side_effect=FileNotFoundError)
    client.write_file = AsyncMock()
    client.settings = s
    task = TaskModel(id="1", description="t", files=files or [])
    return WorkerAgent(task=task, client=client, settings=s), client


def test_parse_file_blocks():
    resp = "FILE: a.py\nprint('hi')\nENDFILE\n"
    matches = list(_FILE_BLOCK_RE.finditer(resp))
    assert len(matches) == 1
    assert matches[0].group("path").strip() == "a.py"
    assert "print('hi')" in matches[0].group("content")


def test_files_written():
    resp = "FILE: a.py\ncontent\nENDFILE\n"
    w, client = make_worker(resp)
    out = asyncio.get_event_loop().run_until_complete(w.run())
    client.write_file.assert_awaited_once()
    assert "Modified 1 file(s)" in out


def test_no_file_blocks():
    resp = "just text"
    w, _client = make_worker(resp)
    out = asyncio.get_event_loop().run_until_complete(w.run())
    assert "No file changes detected" in out
