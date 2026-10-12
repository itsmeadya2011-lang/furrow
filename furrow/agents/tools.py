from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any

import aiofiles

from furrow.llm import LLMClient


class Tool:
    """Base class for tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["path"]
    }

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def execute(self, path: str) -> str:
        try:
            content = await self.client.read_file(path)
            return f"File: {path}\n```\n{content}\n```"
        except Exception as e:
            return f"Error reading {path}: {e}"


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file (creates directories as needed)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write to the file"}
        },
        "required": ["path", "content"]
    }

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def execute(self, path: str, content: str) -> str:
        try:
            await self.client.write_file(path, content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files in a directory"
    parameters = {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory to list (default: current directory)"}
        },
        "required": ["directory"]
    }

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def execute(self, directory: str = ".") -> str:
        try:
            files = self.client.list_files(directory)
            if not files:
                return f"No files found in {directory}"
            return f"Files in {directory}:\n" + "\n".join(f"  {f}" for f in files)
        except Exception as e:
            return f"Error listing {directory}: {e}"


class RunCommandTool(Tool):
    name = "run_command"
    description = "Run a shell command and return its output"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)", "default": 60},
            "cwd": {"type": "string", "description": "Working directory (default: current directory)"}
        },
        "required": ["command"]
    }

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    async def execute(self, command: str, timeout: int = 60, cwd: str = ".") -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode() + stderr.decode()
                if proc.returncode != 0:
                    return f"Command failed (exit code {proc.returncode}):\n{output}"
                return output or "(no output)"
            except asyncio.TimeoutError:
                proc.kill()
                return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error running command: {e}"


class EditFileTool(Tool):
    name = "edit_file"
    description = "Make a targeted edit to a file by replacing a specific string"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "old_string": {"type": "string", "description": "The exact string to replace (must be unique in the file)"},
            "new_string": {"type": "string", "description": "The string to replace it with"}
        },
        "required": ["path", "old_string", "new_string"]
    }

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def execute(self, path: str, old_string: str, new_string: str) -> str:
        try:
            content = await self.client.read_file(path)
            if old_string not in content:
                return f"Error: old_string not found in {path}"
            if content.count(old_string) > 1:
                return f"Error: old_string found multiple times in {path}, be more specific"
            new_content = content.replace(old_string, new_string)
            await self.client.write_file(path, new_content)
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing {path}: {e}"


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"},
            "root": {"type": "string", "description": "Root directory (default: current directory)"}
        },
        "required": ["pattern"]
    }

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    async def execute(self, pattern: str, root: str = ".") -> str:
        try:
            import glob
            matches = glob.glob(os.path.join(root, pattern), recursive=True)
            if not matches:
                return f"No files match pattern: {pattern}"
            rel_matches = [os.path.relpath(m, root) for m in matches]
            return f"Matches for {pattern}:\n" + "\n".join(f"  {m}" for m in rel_matches)
        except Exception as e:
            return f"Error running glob: {e}"


class GrepTool(Tool):
    name = "grep"
    description = "Search for a pattern in files"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "include": {"type": "string", "description": "File pattern to include (e.g., *.py)"},
            "root": {"type": "string", "description": "Root directory (default: current directory)"}
        },
        "required": ["pattern"]
    }

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    async def execute(self, pattern: str, include: str = "*", root: str = ".") -> str:
        try:
            import subprocess
            cmd = ["rg", "--no-heading", "--line-number", pattern]
            if include != "*":
                cmd.extend(["-g", include])
            cmd.append(root)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode() + stderr.decode()
            if proc.returncode == 1:
                return f"No matches for pattern: {pattern}"
            if proc.returncode != 0:
                return f"Grep error: {output}"
            return output or "(no matches)"
        except FileNotFoundError:
            return "Error: ripgrep (rg) not installed"
        except Exception as e:
            return f"Error running grep: {e}"


TOOLS: dict[str, type[Tool]] = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "list_files": ListFilesTool,
    "run_command": RunCommandTool,
    "edit_file": EditFileTool,
    "glob": GlobTool,
    "grep": GrepTool,
}


def get_tool_descriptions() -> str:
    """Get formatted tool descriptions for the prompt."""
    lines = []
    for name, tool_cls in TOOLS.items():
        tool = tool_cls(None)  # type: ignore
        params = tool.parameters
        props = params.get("properties", {})
        required = params.get("required", [])
        param_strs = []
        for pname, pinfo in props.items():
            req = " (required)" if pname in required else ""
            param_strs.append(f"  {pname}: {pinfo.get('description', '')}{req}")
        lines.append(f"{name}:\n  {tool.description}\n" + "\n".join(param_strs))
    return "\n\n".join(lines)


def create_tool(name: str, client: LLMClient) -> Tool:
    """Create a tool instance by name."""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOLS[name](client)