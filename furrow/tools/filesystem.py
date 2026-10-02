from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import aiofiles

from furrow.tools.base import Tool, ToolResult


class ReadFileTool(Tool):
    """Read the content of a file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return """Read the content of a file at the given path. Returns the file content and size.
Useful for examining source code, configs, or any text file."""

    @property
    def _properties(self) -> dict[str, Any]:
        return {
            "path": {
                "type": "string",
                "description": "The path to the file to read, relative to the workspace root.",
            }
        }

    @property
    def _required(self) -> list[str]:
        return ["path"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(success=False, output="", error="Missing 'path' argument")
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        if not p.is_file():
            return ToolResult(success=False, output="", error=f"Not a file: {path}")
        try:
            async with aiofiles.open(p, "r") as f:
                content = await f.read()
            size = p.stat().st_size
            return ToolResult(
                success=True,
                output=f"[File: {path} ({size} bytes)]\n```\n{content}\n```",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Error reading file: {e}")


class WriteFileTool(Tool):
    """Write content to a file, creating directories as needed."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        """Write content to a file at the given path. Creates parent directories if needed.
Overwrites the file if it already exists."""

    @property
    def _properties(self) -> dict[str, Any]:
        return {
            "path": {
                "type": "string",
                "description": "The path to the file to write, relative to the workspace root.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
        }

    @property
    def _required(self) -> list[str]:
        return ["path", "content"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content")
        if not path or content is None:
            return ToolResult(success=False, output="", error="Missing 'path' or 'content' argument")
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)
            return ToolResult(
                success=True,
                output=f"File written successfully: {path}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Error writing file: {e}")


class ListFilesTool(Tool):
    """List files in a directory recursively."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        """List files in a directory, recursively. Returns relative file paths."""

    @property
    def _properties(self) -> dict[str, Any]:
        return {
            "directory": {
                "type": "string",
                "description": "The directory to list files from, relative to the workspace root. "
                "Defaults to current directory if omitted.",
            }
        }

    @property
    def _required(self) -> list[str]:
        return []

    async def execute(self, **kwargs: Any) -> ToolResult:
        directory = kwargs.get("directory", ".")
        p = Path(directory)
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {directory}")
        files: list[str] = []
        for root, _dirs, filenames in os.walk(p):
            for fname in filenames:
                if fname.startswith("."):
                    continue
                filepath = os.path.join(root, fname)
                rel = os.path.relpath(filepath, p)
                files.append(rel)
        files.sort()
        return ToolResult(
            success=True,
            output=f"[{len(files)} files in {directory}]\n" + "\n".join(files),
        )


class GrepTool(Tool):
    """Search for a pattern in files."""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        """Search for a pattern (regex) in files. Returns matching lines with file paths and line numbers.
Searches the workspace directory recursively."""

    @property
    def _properties(self) -> dict[str, Any]:
        return {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for.",
            },
            "directory": {
                "type": "string",
                "description": "The directory to search in. Defaults to current directory.",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for files to include (e.g. '*.py'). Defaults to all files.",
            },
        }

    @property
    def _required(self) -> list[str]:
        return ["pattern"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern")
        directory = kwargs.get("directory", ".")
        file_pattern = kwargs.get("file_pattern", "*")
        if not pattern:
            return ToolResult(success=False, output="", error="Missing 'pattern' argument")
        p = Path(directory)
        if not p.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {directory}")
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")
        matches: list[str] = []
        for filepath in p.rglob(file_pattern):
            if not filepath.is_file():
                continue
            if filepath.suffix == ".pyc" or "/.git/" in str(filepath):
                continue
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel = filepath.relative_to(p)
                        matches.append(f"{rel}:{i}: {line.rstrip()}")
            except Exception:
                continue
        if not matches:
            return ToolResult(success=True, output=f"No matches for pattern '{pattern}'")
        return ToolResult(
            success=True,
            output=f"[{len(matches)} matches]\n" + "\n".join(matches[:200]),
        )


__all__ = [
    "GrepTool",
    "ListFilesTool",
    "ReadFileTool",
    "WriteFileTool",
]
