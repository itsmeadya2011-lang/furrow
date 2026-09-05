from __future__ import annotations

import asyncio
import re
from pathlib import Path

from furrow.tools.base import Tool, ToolResult


class ReadFile(Tool):
    name = "read_file"
    description = "Read the contents of a file. Returns the file contents or an error."

    @classmethod
    async def execute(cls, path: str, line_start: int = 0, line_end: int | None = None) -> ToolResult:
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        try:
            # Run blocking file I/O in a thread to stay async-friendly
            content = await asyncio.to_thread(_read_file_sync, p, line_start, line_end)
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def _read_file_sync(p: Path, line_start: int, line_end: int | None) -> str:
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if line_start > 0:
        lines = lines[line_start:]
    if line_end is not None:
        end = line_end - (line_start if line_start > 0 else 0)
        lines = lines[:end]
    return "".join(lines)


class WriteFile(Tool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed. Overwrites existing files."

    @classmethod
    async def execute(cls, path: str, content: str) -> ToolResult:
        p = Path(path)
        try:
            await asyncio.to_thread(_write_file_sync, p, content)
            return ToolResult(output=f"File written: {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def _write_file_sync(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)


class EditFile(Tool):
    """Find-and-replace within a file. If ``old_str`` is not unique, specify more context."""

    name = "edit_file"
    description = "Edit a file by replacing old_str with new_str. old_str must appear exactly once."

    @classmethod
    async def execute(cls, path: str, old_str: str, new_str: str) -> ToolResult:
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        try:
            result = await asyncio.to_thread(_edit_file_sync, p, old_str, new_str)
            return result
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def _edit_file_sync(p: Path, old_str: str, new_str: str) -> ToolResult:
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    count = content.count(old_str)
    if count == 0:
        return ToolResult(success=False, error=f"old_str not found in {p}")
    if count > 1:
        return ToolResult(
            success=False,
            error=f"old_str found {count} times in {p}. Provide more context to make it unique.",
        )
    content = content.replace(old_str, new_str, 1)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return ToolResult(output=f"File edited: {p}")


class ListFiles(Tool):
    name = "list_files"
    description = "List files in a directory tree. Returns relative file paths."

    @classmethod
    async def execute(cls, path: str = ".", recursive: bool = True) -> ToolResult:
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, error=f"Directory not found: {path}")
        try:
            result = await asyncio.to_thread(_list_files_sync, p, recursive)
            return result
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def _list_files_sync(p: Path, recursive: bool) -> ToolResult:
    if recursive:
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
    else:
        files = [str(f.name) for f in p.iterdir() if f.is_file()]
    return ToolResult(output="\n".join(sorted(files)) if files else "(empty)")


class GrepFiles(Tool):
    name = "grep_files"
    description = "Search for a regex pattern in files. Returns matching lines with file paths."

    @classmethod
    async def execute(cls, pattern: str, path: str = ".", file_pattern: str = "*") -> ToolResult:
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        try:
            result = await asyncio.to_thread(_grep_files_sync, p, pattern, file_pattern)
            return result
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def _grep_files_sync(p: Path, pattern: str, file_pattern: str) -> ToolResult:
    regex = re.compile(pattern)
    results: list[str] = []
    search_root = p if p.is_dir() else p.parent
    for f in _iter_files(search_root, file_pattern):
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if regex.search(line):
                        rel = str(f.relative_to(search_root))
                        results.append(f"{rel}:{i}: {line.rstrip()}")
        except Exception:
            continue
    return ToolResult(output="\n".join(results) if results else "(no matches)")


def _iter_files(root: Path, pattern: str):
    if root.is_dir():
        yield from (f for f in root.rglob(pattern) if f.is_file())
    elif root.is_file():
        yield root
