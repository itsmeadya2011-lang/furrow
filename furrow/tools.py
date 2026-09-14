from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    tool_name: str
    input: dict[str, Any]
    output: str
    is_error: bool = False


async def read_file(path: str, workspace: Path | None = None) -> str:
    """Read a file from the workspace."""
    workspace = workspace or Path.cwd()
    file_path = workspace / path
    
    try:
        content = file_path.read_text(encoding="utf-8")
        return content
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied reading: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


async def write_file(path: str, content: str, workspace: Path | None = None) -> str:
    """Write content to a file. Creates parent directories as needed."""
    workspace = workspace or Path.cwd()
    file_path = workspace / path
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"
    except PermissionError:
        return f"Error: Permission denied writing to: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def edit_file(path: str, old_text: str, new_text: str, workspace: Path | None = None) -> str:
    """Replace old_text with new_text in a file."""
    workspace = workspace or Path.cwd()
    file_path = workspace / path
    
    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"
    
    if old_text not in content:
        return f"Error: Old text not found in {path}"
    
    new_content = content.replace(old_text, new_text)
    
    try:
        file_path.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def list_files(directory: str = ".", workspace: Path | None = None) -> list[str]:
    """List all files recursively in a directory."""
    workspace = workspace or Path.cwd()
    dir_path = workspace / directory
    
    try:
        files = []
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(workspace)
                files.append(str(rel_path))
        return sorted(files)
    except FileNotFoundError:
        return [f"Error: Directory not found: {directory}"]
    except Exception as e:
        return [f"Error listing files: {e}"]


async def run_command(command: str, workspace: Path | None = None, timeout: int = 60) -> str:
    """Execute a shell command with timeout."""
    workspace = workspace or Path.cwd()
    
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"Error: Command timed out after {timeout}s"
        
        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            output += stderr.decode("utf-8", errors="replace")
        
        if process.returncode != 0:
            output = f"Exit code: {process.returncode}\n{output}"
        
        return output
    except Exception as e:
        return f"Error running command: {e}"


async def grep_files(pattern: str, directory: str = ".", workspace: Path | None = None) -> list[str]:
    """Search for a regex pattern in files."""
    workspace = workspace or Path.cwd()
    dir_path = workspace / directory
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [f"Error: Invalid regex pattern: {e}"]
    
    matches = []
    
    try:
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError):
                    continue
                
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel_path = file_path.relative_to(workspace)
                        matches.append(f"{rel_path}:{i}: {line}")
        
        return matches
    except FileNotFoundError:
        return [f"Error: Directory not found: {directory}"]
    except Exception as e:
        return [f"Error searching files: {e}"]


ANTHROPIC_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the workspace",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace old_text with new_text in a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "old_text": {"type": "string", "description": "Text to replace"},
                "new_text": {"type": "string", "description": "New text to insert"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files recursively in a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path relative to workspace", "default": "."},
            },
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": "Execute a shell command (for running tests, lint, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
            },
            "required": ["command"],
        },
    },
    {
        "name": "grep_files",
        "description": "Search for a regex pattern in files",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "directory": {"type": "string", "description": "Directory path relative to workspace", "default": "."},
            },
            "required": ["pattern"],
        },
    },
]


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace old_text with new_text in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "old_text": {"type": "string", "description": "Text to replace"},
                    "new_text": {"type": "string", "description": "New text to insert"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files recursively in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path relative to workspace", "default": "."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command (for running tests, lint, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Search for a regex pattern in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "directory": {"type": "string", "description": "Directory path relative to workspace", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
]