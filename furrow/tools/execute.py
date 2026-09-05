from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

from furrow.tools.base import Tool, ToolResult


class ExecuteCommand(Tool):
    """Run a shell command and return its combined stdout/stderr.

    Executes in the current working directory. Commands are run via
    ``asyncio.create_subprocess_exec`` so they are fully async-safe
    inside the agent loop.
    """

    name = "execute_command"
    description = "Execute a shell command and return its output. Use this for running tests, linters, installing packages, etc."

    @classmethod
    async def execute(cls, command: str, timeout: int = 120, cwd: str | None = None) -> ToolResult:
        run_dir = Path(cwd) if cwd else Path.cwd()
        try:
            parts = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(run_dir),
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                returncode = proc.returncode or 0
                output = stdout.decode("utf-8", errors="replace")
                return ToolResult(
                    success=returncode == 0,
                    output=output,
                    error=None if returncode == 0 else f"Exit code: {returncode}",
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(success=False, error=f"Command timed out after {timeout}s")
        except FileNotFoundError:
            return ToolResult(success=False, error=f"Command not found: {shlex.split(command)[0]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
