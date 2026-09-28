from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from furrow.llm import ToolCall

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file at the given path within the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file to read."}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, overwriting it if it exists. Creates parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "The full content to write."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace the first occurrence of old_string with new_string in the given file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit."},
                "old_string": {"type": "string", "description": "The string to replace."},
                "new_string": {"type": "string", "description": "The replacement string."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files (recursively) under the given path as a JSON array of relative paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Defaults to current directory.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the workspace and return its combined stdout/stderr output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds. Defaults to 60.",
                    "default": 60,
                },
            },
            "required": ["command"],
        },
    },
]


class ToolRunner:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)

    def get_tool_names(self) -> list[str]:
        return [s["name"] for s in TOOL_SCHEMAS]

    async def execute(self, call: ToolCall) -> str:
        fn = getattr(self, f"_tool_{call.name}", None)
        if fn is None:
            return f"Error: unknown tool '{call.name}'"
        try:
            result = await fn(**call.args)
            if isinstance(result, str):
                return result
            return json.dumps(result)
        except Exception as e:  # noqa: BLE001  # report any tool error
            return f"Error: {type(e).__name__}: {e}"

    async def _tool_read_file(self, path: str) -> str:
        p = self._resolve(path)
        return p.read_text()

    async def _tool_write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {p}"

    async def _tool_edit_file(self, path: str, old_string: str, new_string: str) -> str:
        p = self._resolve(path)
        content = p.read_text()
        if old_string not in content:
            raise ValueError("old_string not found in file")
        count = content.count(old_string)
        content = content.replace(old_string, new_string, 1)
        p.write_text(content)
        return f"Edited {p} (matched {count} occurrence(s))"

    async def _tool_list_files(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.exists():
            return json.dumps([])
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        return json.dumps(files)

    async def _tool_run_command(self, command: str, timeout: float = 60) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
        )
        transport = getattr(proc, "_transport", None)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            await proc.wait()
            if transport is not None and not transport.is_closing():
                transport.close()
            return (out.decode(errors="replace") + err.decode(errors="replace")).strip()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            if transport is not None and not transport.is_closing():
                transport.close()
            return f"Timeout after {timeout}s"

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        return p.resolve()
