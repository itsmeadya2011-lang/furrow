from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from furrow.llm import LLMClient

DEFAULT_TIMEOUT = 30


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "json",
            "properties": properties,
            "required": required or list(properties.keys()),
        },
    }


TOOLS: list[dict[str, Any]] = [
    _schema(
        "read_file",
        "Read the entire contents of a file at the given path.",
        {"path": {"type": "string", "description": "Absolute or relative path to the file."}},
    ),
    _schema(
        "write_file",
        "Write content to a file at the given path, creating parent directories as needed.",
        {
            "path": {"type": "string", "description": "Absolute or relative path to the file."},
            "content": {"type": "string", "description": "The text content to write to the file."},
        },
    ),
    _schema(
        "list_files",
        "List all files under a directory recursively as paths relative to it.",
        {"directory": {"type": "string", "description": "Absolute or relative directory path."}},
    ),
    _schema(
        "search_text",
        "Search for a substring in files under a directory; returns matches with line numbers.",
        {
            "pattern": {"type": "string", "description": "The substring to search for."},
            "directory": {"type": "string", "description": "Absolute or relative directory path to search."},
        },
    ),
    _schema(
        "run_shell",
        "Run a shell command via bash and return its stdout, stderr, and return code.",
        {
            "command": {"type": "string", "description": "The shell command to execute."},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds before the command is killed.",
                "default": 30,
            },
        },
        required=["command"],
    ),
]


def openai_tools() -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in TOOLS:
        schema = dict(tool["input_schema"])
        schema["type"] = "object"
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": schema,
                },
            }
        )
    return converted


def _resolve(path: str, workspace: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (workspace / p)


async def read_file(path: str, workspace: Path, client: LLMClient | None = None) -> dict[str, Any]:
    try:
        resolved = _resolve(path, workspace)
        content = await _client(client).read_file(resolved)
        return {"result": content}
    except Exception as e:
        return {"error": str(e)}


async def write_file(path: str, content: str, workspace: Path, client: LLMClient | None = None) -> dict[str, Any]:
    try:
        resolved = _resolve(path, workspace)
        await _client(client).write_file(resolved, content)
        return {"result": f"Successfully wrote to {path}"}
    except Exception as e:
        return {"error": str(e)}


async def list_files(directory: str, workspace: Path, client: LLMClient | None = None) -> dict[str, Any]:
    try:
        resolved = _resolve(directory, workspace)
        files = _client(client).list_files(resolved)
        return {"result": files}
    except Exception as e:
        return {"error": str(e)}


async def search_text(
    pattern: str,
    directory: str,
    workspace: Path,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    try:
        resolved = _resolve(directory, workspace)
        matches: list[str] = []
        for f in resolved.rglob("*"):
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    matches.append(f"{f}:{i}: {line}")
                    if len(matches) >= 100:
                        return {"result": matches}
        return {"result": matches}
    except Exception as e:
        return {"error": str(e)}


async def run_shell(
    command: str,
    workspace: Path,
    timeout: int = DEFAULT_TIMEOUT,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        "bash", "-c", command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"error": "Command timed out", "stdout": "", "stderr": "", "returncode": 124}
    return {
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "returncode": proc.returncode,
    }


def _client(client: LLMClient | None) -> LLMClient:
    if client is not None:
        return client
    from furrow.config import settings

    return LLMClient(settings=settings)


async def execute_tool(name: str, args: dict[str, Any], workspace: Path) -> dict[str, Any]:
    ws = Path(workspace)
    try:
        if name == "read_file":
            return await read_file(args["path"], ws)
        elif name == "write_file":
            return await write_file(args["path"], args["content"], ws)
        elif name == "list_files":
            return await list_files(args["directory"], ws)
        elif name == "search_text":
            return await search_text(args["pattern"], args["directory"], ws)
        elif name == "run_shell":
            return await run_shell(args["command"], ws, timeout=args.get("timeout", DEFAULT_TIMEOUT))
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}
