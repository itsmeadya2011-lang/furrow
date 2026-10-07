from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

console = Console()


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


class Tool(ABC):
    """Base class for tools."""

    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        pass

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
        }

    @abstractmethod
    def get_parameters_schema(self) -> dict:
        pass


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            if not p.is_file():
                return ToolResult(success=False, output="", error=f"Not a file: {path}")

            content = p.read_text()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file (creates directories as needed)"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return ToolResult(success=True, output=f"Written {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class EditFileTool(Tool):
    name = "edit_file"
    description = "Make targeted edits to a file using exact string replacement"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {"type": "string", "description": "Exact string to replace"},
                "new_string": {"type": "string", "description": "New string to replace with"},
            },
            "required": ["path", "old_string", "new_string"],
        }

    async def execute(self, path: str, old_string: str, new_string: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")

            content = p.read_text()
            if old_string not in content:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Old string not found in file. File length: {len(content)} chars",
                )

            new_content = content.replace(old_string, new_string)
            if new_content == content:
                return ToolResult(success=False, output="", error="No changes made (strings identical)")

            p.write_text(new_content)
            return ToolResult(success=True, output=f"Edited {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files in a directory"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current directory)"},
                "pattern": {"type": "string", "description": "Glob pattern to filter files"},
            },
            "required": [],
        }

    async def execute(self, path: str = ".", pattern: str = "**/*") -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"Directory not found: {path}")

            files = []
            for f in p.rglob(pattern):
                if f.is_file():
                    rel = f.relative_to(p)
                    files.append(str(rel))

            return ToolResult(success=True, output="\n".join(sorted(files)) if files else "No files found")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class RunCommandTool(Tool):
    name = "run_command"
    description = "Run a shell command and return output"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"},
                "cwd": {"type": "string", "description": "Working directory (default: current)"},
                "timeout": {"type": "number", "description": "Timeout in seconds (default: 60)"},
            },
            "required": ["command"],
        }

    async def execute(self, command: str, cwd: str = ".", timeout: int = 60) -> ToolResult:
        import asyncio

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode() + stderr.decode()
                if proc.returncode == 0:
                    return ToolResult(success=True, output=output)
                else:
                    return ToolResult(success=False, output=output, error=f"Exit code: {proc.returncode}")
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(success=False, output="", error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"},
                "path": {"type": "string", "description": "Base directory (default: current)"},
            },
            "required": ["pattern"],
        }

    async def execute(self, pattern: str, path: str = ".") -> ToolResult:
        try:
            import glob

            p = Path(path)
            full_pattern = str(p / pattern)
            matches = glob.glob(full_pattern, recursive=True)
            rel_matches = [str(Path(m).relative_to(p)) for m in matches]
            return ToolResult(success=True, output="\n".join(sorted(rel_matches)) if rel_matches else "No matches")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class GrepTool(Tool):
    name = "grep"
    description = "Search for text in files"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search (default: current)"},
                "include": {"type": "string", "description": "File pattern to include (e.g., *.py)"},
            },
            "required": ["pattern"],
        }

    async def execute(self, pattern: str, path: str = ".", include: str = "*") -> ToolResult:
        try:
            import re

            p = Path(path)
            results = []
            regex = re.compile(pattern)

            for f in p.rglob(include):
                if f.is_file():
                    try:
                        content = f.read_text()
                        for i, line in enumerate(content.splitlines(), 1):
                            if regex.search(line):
                                rel = f.relative_to(p)
                                results.append(f"{rel}:{i}: {line.strip()}")
                    except Exception:
                        continue

            return ToolResult(success=True, output="\n".join(results) if results else "No matches")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


# Tool registry
TOOLS: dict[str, Tool] = {
    "read_file": ReadFileTool(),
    "write_file": WriteFileTool(),
    "edit_file": EditFileTool(),
    "list_files": ListFilesTool(),
    "run_command": RunCommandTool(),
    "glob": GlobTool(),
    "grep": GrepTool(),
}


def get_tool(name: str) -> Optional[Tool]:
    return TOOLS.get(name)


def get_all_tools() -> list[Tool]:
    return list(TOOLS.values())


def get_tool_schemas() -> list[dict]:
    return [tool.get_schema() for tool in TOOLS.values()]


async def execute_tool(name: str, **kwargs: Any) -> ToolResult:
    tool = get_tool(name)
    if not tool:
        return ToolResult(success=False, output="", error=f"Unknown tool: {name}")
    return await tool.execute(**kwargs)