from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles


class ToolError(Exception):
    pass


class Tool:
    """Base class for tools."""
    name: str = ""
    description: str = ""
    
    async def execute(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file"
    
    async def execute(self, path: str) -> str:
        path = Path(path)
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if not path.is_file():
            raise ToolError(f"Not a file: {path}")
        async with aiofiles.open(path, "r") as f:
            return await f.read()


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file (creates parent directories if needed)"
    
    async def execute(self, path: str, content: str) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w") as f:
            await f.write(content)
        return f"Written to {path}"


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files in a directory recursively"
    
    async def execute(self, directory: str = ".") -> list[str]:
        path = Path(directory)
        if not path.exists():
            raise ToolError(f"Directory not found: {path}")
        if not path.is_dir():
            raise ToolError(f"Not a directory: {path}")
        return [str(f.relative_to(path)) for f in path.rglob("*") if f.is_file()]


class RunCommandTool(Tool):
    name = "run_command"
    description = "Run a shell command and return stdout/stderr"
    
    async def execute(self, command: list[str], cwd: str | None = None, timeout: int = 120) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return {
                    "returncode": proc.returncode,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                }
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise ToolError(f"Command timed out after {timeout}s: {' '.join(command)}")
        except FileNotFoundError:
            raise ToolError(f"Command not found: {command[0]}")
        except Exception as e:
            raise ToolError(f"Failed to run command: {e}")


class EditFileTool(Tool):
    name = "edit_file"
    description = "Make a targeted edit to a file by replacing old_string with new_string"
    
    async def execute(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        path = Path(path)
        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if not path.is_file():
            raise ToolError(f"Not a file: {path}")
        
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        
        if old_string not in content:
            raise ToolError(f"Old string not found in file: {old_string[:50]}...")
        
        if not replace_all and content.count(old_string) > 1:
            raise ToolError(f"Old string found multiple times ({content.count(old_string)}). Use replace_all=True or provide more context.")
        
        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        
        async with aiofiles.open(path, "w") as f:
            await f.write(new_content)
        
        return f"Edited {path}"


TOOLS: dict[str, Tool] = {
    "read_file": ReadFileTool(),
    "write_file": WriteFileTool(),
    "list_files": ListFilesTool(),
    "run_command": RunCommandTool(),
    "edit_file": EditFileTool(),
}


def get_tool(name: str) -> Tool | None:
    return TOOLS.get(name)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI/Anthropic compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (creates parent directories if needed)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory recursively",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory to list (default: current directory)"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command and return stdout/stderr",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "array", "items": {"type": "string"}, "description": "Command and arguments to run"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Make a targeted edit to a file by replacing old_string with new_string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "old_string": {"type": "string", "description": "String to replace (must be unique or use replace_all)"},
                        "new_string": {"type": "string", "description": "String to replace with"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false)"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
    ]