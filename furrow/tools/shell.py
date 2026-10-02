from __future__ import annotations

import asyncio
from typing import Any

from furrow.tools.base import Tool, ToolResult


class ShellTool(Tool):
    """Execute a shell command and return its output."""

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        """Execute a shell command in the workspace and return stdout/stderr.
Use this for running tests, installing dependencies, building, checking git status, etc.
The command runs in a subshell with the workspace as the working directory."""

    @property
    def _properties(self) -> dict[str, Any]:
        return {
            "command": {
                "type": "string",
                "description": "The shell command to execute. Use && or ; to chain commands.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds for the command. Default 60, max 300.",
                "default": 60,
            },
        }

    @property
    def _required(self) -> list[str]:
        return ["command"]

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command")
        if not command:
            return ToolResult(success=False, output="", error="Missing 'command' argument")
        timeout = min(kwargs.get("timeout", 60), 300)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout}s: {command}",
                )
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            returncode = proc.returncode or 0
            parts: list[str] = []
            if stdout:
                parts.append(f"[stdout]\n{stdout}")
            if stderr:
                parts.append(f"[stderr]\n{stderr}")
            parts.append(f"[exit code: {returncode}]")
            output = "\n".join(parts)
            success = returncode == 0
            return ToolResult(
                success=success,
                output=output,
                error=stderr if not success and stderr else None,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error=f"Command not found: {command.split()[0]}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error executing command: {e}",
            )


__all__ = ["ShellTool"]
