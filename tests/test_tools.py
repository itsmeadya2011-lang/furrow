import asyncio
from pathlib import Path

import pytest

from furrow.tools import (
    edit_file,
    grep_files,
    list_files,
    read_file,
    run_command,
    write_file,
)


class TestReadFile:
    """Tests for the read_file tool."""

    async def test_read_existing_file(self, tmp_path: Path) -> None:
        """Reading an existing file returns its contents."""
        file_path = tmp_path / "hello.txt"
        file_path.write_text("Hello, world!", encoding="utf-8")

        result = await read_file("hello.txt", workspace=tmp_path)
        assert result == "Hello, world!"

    async def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Reading a non-existent file returns an error message."""
        result = await read_file("missing.txt", workspace=tmp_path)
        assert "Error: File not found" in result


class TestWriteFile:
    """Tests for the write_file tool."""

    async def test_write_new_file(self, tmp_path: Path) -> None:
        """Writing to a new file creates it with the given content."""
        result = await write_file("new.txt", "fresh content", workspace=tmp_path)
        assert "Successfully wrote" in result
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "fresh content"

    async def test_write_overwrite_existing(self, tmp_path: Path) -> None:
        """Overwriting an existing file replaces its contents."""
        (tmp_path / "existing.txt").write_text("old content", encoding="utf-8")
        result = await write_file("existing.txt", "new content", workspace=tmp_path)
        assert "Successfully wrote" in result
        assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "new content"


class TestEditFile:
    """Tests for the edit_file tool."""

    async def test_edit_file_success(self, tmp_path: Path) -> None:
        """Editing replaces old_text with new_text."""
        (tmp_path / "doc.txt").write_text("Hello world", encoding="utf-8")
        result = await edit_file("doc.txt", "world", "pytest", workspace=tmp_path)
        assert "Successfully edited" in result
        assert (tmp_path / "doc.txt").read_text(encoding="utf-8") == "Hello pytest"

    async def test_edit_file_not_found(self, tmp_path: Path) -> None:
        """Editing a non-existent file returns an error."""
        result = await edit_file("missing.txt", "old", "new", workspace=tmp_path)
        assert "Error: File not found" in result

    async def test_edit_file_old_text_not_found(self, tmp_path: Path) -> None:
        """Editing when old_text is not in the file returns an error."""
        (tmp_path / "doc.txt").write_text("Hello world", encoding="utf-8")
        result = await edit_file("doc.txt", "goodbye", "pytest", workspace=tmp_path)
        assert "Error: Old text not found" in result


class TestListFiles:
    """Tests for the list_files tool."""

    async def test_list_files(self, tmp_path: Path) -> None:
        """Listing files returns all files recursively."""
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")
        result = await list_files(".", workspace=tmp_path)
        assert sorted(result) == ["a.txt", "sub/b.txt"]


class TestRunCommand:
    """Tests for the run_command tool."""

    async def test_run_command_success(self, tmp_path: Path) -> None:
        """Running a command with zero exit code returns its output."""
        result = await run_command("echo hello", workspace=tmp_path)
        assert "hello" in result

    async def test_run_command_failure(self, tmp_path: Path) -> None:
        """Running a command with non-zero exit code returns error info."""
        result = await run_command("exit 1", workspace=tmp_path)
        assert "Exit code: 1" in result


class TestGrepFiles:
    """Tests for the grep_files tool."""

    async def test_grep_finds_pattern(self, tmp_path: Path) -> None:
        """Searching for a pattern returns matching lines."""
        (tmp_path / "code.py").write_text("x = 1\ny = 2\nx = 3", encoding="utf-8")
        result = await grep_files("x", workspace=tmp_path)
        assert len(result) == 2
        assert any("code.py" in r for r in result)

    async def test_grep_no_matches(self, tmp_path: Path) -> None:
        """Searching for a non-existent pattern returns empty list."""
        (tmp_path / "code.py").write_text("hello", encoding="utf-8")
        result = await grep_files("nomatch", workspace=tmp_path)
        assert result == []