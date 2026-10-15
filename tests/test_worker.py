from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class FakeClient:
    def __init__(self, workspace: Path) -> None:
        self.written: dict[str, str] = {}
        self.settings = SimpleNamespace(workspace=workspace)

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[str(path)] = content


def make_worker(client: FakeClient) -> WorkerAgent:
    task = TaskModel(id="1", description="test task", files=["src/foo.py"])
    return WorkerAgent(task=task, client=client)


def test_parse_single_file() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = "FILE: src/foo.py\n```python\nprint('hello')\n```\n"
    changes = worker._parse_file_changes(response)
    assert changes == [("src/foo.py", "print('hello')\n")]


def test_parse_multiple_files() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = (
        "FILE: src/a.py\n```python\nx = 1\n```\n"
        "FILE: src/b.py\n```python\ny = 2\n```\n"
    )
    changes = worker._parse_file_changes(response)
    assert len(changes) == 2
    assert changes[0] == ("src/a.py", "x = 1\n")
    assert changes[1] == ("src/b.py", "y = 2\n")


def test_parse_fence_without_language_tag() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = "FILE: config.yaml\n```\nkey: value\n```\n"
    changes = worker._parse_file_changes(response)
    assert changes == [("config.yaml", "key: value\n")]


def test_parse_no_space_after_colon() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = "FILE:src/foo.py\n```python\nx = 1\n```\n"
    changes = worker._parse_file_changes(response)
    assert changes == [("src/foo.py", "x = 1\n")]


def test_parse_backtick_wrapped_path() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = "FILE: `src/foo.py`\n```python\nx = 1\n```\n"
    changes = worker._parse_file_changes(response)
    assert changes == [("src/foo.py", "x = 1\n")]


def test_parse_whole_response_wrapped_in_fence() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = "```\nFILE: src/foo.py\n```python\nx = 1\n```\n```\n"
    changes = worker._parse_file_changes(response)
    assert changes == [("src/foo.py", "x = 1\n")]


def test_parse_blank_lines_between_marker_and_fence() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = "FILE: src/foo.py\n\n\n```python\nx = 1\n```\n"
    changes = worker._parse_file_changes(response)
    assert changes == [("src/foo.py", "x = 1\n")]


def test_parse_surrounding_text() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = "Here is the change:\n\nFILE: src/foo.py\n```python\nx = 1\n```\nDone.\n"
    changes = worker._parse_file_changes(response)
    assert changes == [("src/foo.py", "x = 1\n")]


def test_parse_empty_response() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    assert worker._parse_file_changes("") == []


def test_parse_no_file_blocks() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    assert worker._parse_file_changes("No files need to be written.\n") == []


def test_parse_content_with_internal_backticks() -> None:
    worker = make_worker(FakeClient(workspace=Path("/tmp/test")))
    response = 'FILE: src/foo.py\n```python\nmsg = "hello"\n```\n'
    changes = worker._parse_file_changes(response)
    assert changes == [("src/foo.py", 'msg = "hello"\n')]


@pytest.mark.asyncio
async def test_apply_changes_relative_to_workspace(tmp_path: Path) -> None:
    client = FakeClient(workspace=tmp_path)
    worker = make_worker(client)
    response = (
        "FILE: src/foo.py\n```python\nprint('hello')\n```\n"
        "FILE: bar.py\n```python\nx = 1\n```\n"
    )
    written = await worker._apply_changes(response)
    assert written == ["src/foo.py", "bar.py"]
    assert (tmp_path / "src/foo.py").read_text() == "print('hello')\n"
    assert (tmp_path / "bar.py").read_text() == "x = 1\n"
    assert client.written[str(tmp_path / "src/foo.py")] == "print('hello')\n"


@pytest.mark.asyncio
async def test_apply_changes_creates_parent_dirs(tmp_path: Path) -> None:
    client = FakeClient(workspace=tmp_path)
    worker = make_worker(client)
    await worker._apply_changes("FILE: a/b/c/deep.py\n```\npass\n```\n")
    assert (tmp_path / "a/b/c/deep.py").read_text() == "pass\n"


@pytest.mark.asyncio
async def test_apply_changes_no_files(tmp_path: Path) -> None:
    worker = make_worker(FakeClient(workspace=tmp_path))
    written = await worker._apply_changes("Nothing to do here.\n")
    assert written == []
