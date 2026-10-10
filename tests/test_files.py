"""Tests for the async file utilities in furrow.files.

These use the tmp_path fixture and asyncio.run() so they work without
pytest-asyncio being installed.
"""

import asyncio
from pathlib import Path

from furrow.files import list_files, read_file, write_file


def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "file.txt"
    asyncio.run(write_file(target, "hello world"))
    assert target.read_text() == "hello world"


def test_read_file_returns_written_content(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    asyncio.run(write_file(target, "payload"))
    assert asyncio.run(read_file(target)) == "payload"


def test_read_file_accepts_string_path(tmp_path: Path) -> None:
    target = tmp_path / "str_path.txt"
    asyncio.run(write_file(str(target), "ok"))
    assert asyncio.run(read_file(str(target))) == "ok"


def test_list_files_returns_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    (tmp_path / "README.md").write_text("# hi")

    files = asyncio.run(list_files(tmp_path))

    assert sorted(files) == ["README.md", "src/app.py"]


def test_list_files_empty_for_nonexistent_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert asyncio.run(list_files(missing)) == []


def test_write_file_overwrites_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "over.txt"
    asyncio.run(write_file(target, "first"))
    asyncio.run(write_file(target, "second"))
    assert asyncio.run(read_file(target)) == "second"