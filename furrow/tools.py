from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from furrow.llm import Tool
from furrow.config import settings


class ToolRegistry:
    """Registry of tools the LLM can invoke (file I/O and shell)."""

    def __init__(self, workspace: Path | None = None, timeout: float = 30.0) -> None:
        self.workspace = workspace or settings.workspace
        self.timeout = timeout

    @property
    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="read_file",
                description="Read a file from the workspace and return its contents as text.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute path",
                        }
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="write_file",
                description="Write content to a file in the workspace, creating parent dirs as needed.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                name="list_files",
                description="List entries in a path within the workspace; optionally walk recursively.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "recursive": {"type": "boolean", "default": False},
                    },
                    "required": [],
                },
            ),
            Tool(
                name="run_bash",
                description="Run a shell command via bash and return combined stdout and stderr.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "number", "default": 30.0},
                    },
                    "required": ["command"],
                },
            ),
        ]

    # -- Handlers (each async, always returns a str) --

    async def read_file(self, path: str) -> str:
        try:
            p = Path(path)
            if not p.is_absolute():
                p = (self.workspace / path).resolve()
            return await asyncio.to_thread(p.read_text)
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    async def write_file(self, path: str, content: str) -> str:
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.workspace / path
            p.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(p.write_text, content)
            return f"Wrote {path} ({len(content)} bytes)"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    async def list_files(self, path: str = ".", recursive: bool = False) -> str:
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.workspace / path
            if not p.exists():
                return f"Error: path does not exist: {path}"
            if recursive:
                entries = sorted(str(f.relative_to(p)) for f in p.rglob("*"))
            else:
                entries = sorted(e.name for e in p.iterdir())
            return "\n".join(entries)
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    async def run_bash(self, command: str, timeout: float = 30.0) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-lc",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:  # noqa: BLE001
                    pass
                return f"Error: command timed out after {timeout}s"
            output = (stdout.decode(errors="replace") if stdout else "") + (
                stderr.decode(errors="replace") if stderr else ""
            )
            return output
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"

    # -- Dispatch --

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return await self.read_file(arguments.get("path", ""))
            if name == "write_file":
                return await self.write_file(
                    arguments.get("path", ""), arguments.get("content", "")
                )
            if name == "list_files":
                return await self.list_files(
                    arguments.get("path", "."), arguments.get("recursive", False)
                )
            if name == "run_bash":
                return await self.run_bash(
                    arguments.get("command", ""),
                    arguments.get("timeout", self.timeout),
                )
            return f"Error: unknown tool '{name}'"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}"
