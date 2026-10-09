from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

MAX_OUTPUT = 10_000

WORKER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content. Creates parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files recursively under a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"directory": {"type": "string", "description": "Directory to scan"}},
            "required": ["directory"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return its combined stdout/stderr. Use for tests, linting, git, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 60, max 300)"},
            },
            "required": ["command"],
        },
    },
]


async def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a worker tool and return a string result (never raises)."""
    try:
        if name == "read_file":
            return _read_file(str(args.get("path", "")))
        if name == "write_file":
            return _write_file(str(args.get("path", "")), str(args.get("content", "")))
        if name == "list_files":
            return _list_files(str(args.get("directory", ".")))
        if name == "run_command":
            return await _run_command(str(args.get("command", "")), int(args.get("timeout") or 60))
        return f"ERROR: unknown tool: {name}"
    except Exception as e:  # noqa: BLE001 - never-raises contract
        return f"ERROR: {type(e).__name__}: {e}"


def _read_file(path: str) -> str:
    return Path(path).read_text()


def _write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"OK: wrote {len(content)} chars to {path}"


def _list_files(directory: str) -> str:
    p = Path(directory)
    if not p.exists():
        return "ERROR: directory not found"
    files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
    return "\n".join(sorted(files)) if files else "(empty)"


async def _run_command(command: str, timeout: int) -> str:
    timeout = max(1, min(300, timeout))
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
        return f"ERROR: command timed out after {timeout}s"
    output = (stdout or b"") + (stderr or b"")
    text = output.decode(errors="replace") if isinstance(output, bytes) else str(output)
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + "... (truncated)"
    return f"EXIT {proc.returncode}\n{text}"