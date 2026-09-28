from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiofiles

import furrow.config
from furrow.config import Settings


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Awaitable[str]]


class ToolRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or furrow.config.settings
        self.base_dir: Path = self.settings.workspace
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            "read_file",
            "Reads the contents of a file. Returns an error message if the file cannot be read.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                },
                "required": ["path"],
            },
            self.read_file,
        )
        self.register(
            "write_file",
            "Writes content to a file, creating parent directories as needed.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write to."},
                    "content": {"type": "string", "description": "Content to write to the file."},
                },
                "required": ["path", "content"],
            },
            self.write_file,
        )
        self.register(
            "list_files",
            "Lists files recursively in a directory, one file per line.",
            {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to list files from. Defaults to the current directory.",
                    },
                },
            },
            self.list_files,
        )
        self.register(
            "grep_files",
            "Searches for a regex pattern in files under a path. Returns matching lines as file:line:content. "
            "Limits results to 100 and skips the .git directory.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Defaults to the current directory.",
                    },
                },
                "required": ["pattern"],
            },
            self.grep_files,
        )
        self.register(
            "run_command",
            "Runs a shell command and returns its combined stdout and stderr output. "
            "Kills the process if it exceeds the timeout.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds. Defaults to 30.0.",
                    },
                },
                "required": ["command"],
            },
            self.run_command,
        )

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Awaitable[str]],
    ) -> None:
        self._tools[name] = Tool(name=name, description=description, parameters=parameters, func=func)

    def get_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_anthropic_tools(self) -> list[dict]:
        return [
            {
                "type": "tool",
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def get_openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Tool '{name}' not found."
        try:
            return await tool.func(**kwargs)
        except Exception as e:
            return f"Error: {e}"

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        return p.resolve()

    async def read_file(self, path: str) -> str:
        try:
            resolved = self._resolve_path(path)
            async with aiofiles.open(resolved, "r") as f:
                return await f.read()
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as e:
            return f"Error: {e}"

    async def write_file(self, path: str, content: str) -> str:
        try:
            resolved = self._resolve_path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(resolved, "w") as f:
                await f.write(content)
            return f"File written: {resolved}"
        except Exception as e:
            return f"Error: {e}"

    async def list_files(self, directory: str = ".") -> str:
        try:
            resolved = self._resolve_path(directory)
            if not resolved.exists():
                return f"Error: Directory not found: {directory}"
            files = []
            for f in resolved.rglob("*"):
                if not f.is_file():
                    continue
                try:
                    files.append(str(f.relative_to(resolved)))
                except ValueError:
                    files.append(str(f))
            files.sort()
            return "\n".join(files)
        except Exception as e:
            return f"Error: {e}"

    async def grep_files(self, pattern: str, path: str = ".") -> str:
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return f"Error: Path not found: {path}"
            regex = re.compile(pattern)
            results: list[str] = []

            if resolved.is_dir():
                targets = [f for f in resolved.rglob("*") if f.is_file() and ".git" not in f.parts]
            else:
                targets = [resolved]

            for f in targets:
                try:
                    async with aiofiles.open(f, "r", encoding="utf-8") as fh:
                        content = await fh.read()
                except Exception:
                    continue
                for lineno, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        try:
                            rel = f.relative_to(resolved)
                        except ValueError:
                            rel = f
                        results.append(f"{rel}:{lineno}:{line}")
                        if len(results) >= 100:
                            return "\n".join(results)
            return "\n".join(results)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        except Exception as e:
            return f"Error: {e}"

    async def run_command(self, command: str, timeout: float = 30.0) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Error: Command timed out after {timeout}s"
            parts = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                parts.append(stderr.decode("utf-8", errors="replace"))
            return "\n".join(parts) if parts else ""
        except Exception as e:
            return f"Error: {e}"


default_tools = ToolRegistry()
