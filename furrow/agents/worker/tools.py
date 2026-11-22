"""Tool implementations for the worker agent.

This module contains the actual implementations of the tools available to the
worker: read_file, write_file, list_files, run_command, git operations, and custom tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from pathlib import Path
from typing import Any

from furrow.agents.custom_tools import execute_custom_tool, get_custom_tools
from furrow.agents.git_tools import execute_git_tool, GIT_TOOLS
from furrow.agents.worker.command_policy import (
    MAX_LIST_ENTRIES,
    MAX_READ_CHARS,
    is_command_denied,
    resolve_in_workspace,
)
from furrow.config import Settings
from furrow.llm import LLMClient, ToolCall


class WorkerTools:
    """Tool implementations for the worker agent."""

    def __init__(
        self,
        client: LLMClient,
        settings: Settings,
        workspace: Path,
    ) -> None:
        self.client = client
        self.settings = settings
        self.workspace = workspace
        self.files_changed: set[str] = set()

    async def execute(self, tc: ToolCall) -> str:
        """Execute a tool call and return the result."""
        name = tc.name
        args = self._coerce_tool_args(tc.arguments)
        if args is None:
            return "error: invalid tool arguments"
        if name == "read_file":
            return await self._read_file(str(args.get("path", "")))
        if name == "write_file":
            if "content" not in args:
                return "error: missing content"
            content = args["content"]
            if not isinstance(content, str):
                return "error: invalid content (expected a string)"
            return await self._write_file(str(args.get("path", "")), content)
        if name == "list_files":
            return await self._list_files(
                str(args.get("path", "")) if args.get("path") is not None else "."
            )
        if name == "run_command":
            return await self._run_command(str(args.get("command", "")))
        # Git tools
        if name in ("git_status", "git_diff", "git_log", "git_branch", "git_commit", "git_checkout"):
            return await execute_git_tool(self.workspace, name, args)
        # Custom tools
        custom_tool_names = {t["name"] for t in get_custom_tools()}
        if name in custom_tool_names:
            return await execute_custom_tool(name, self.workspace, args)
        return f"unknown tool: {name}"

    @staticmethod
    def _coerce_tool_args(arguments: Any) -> dict[str, Any] | None:
        import json
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except (ValueError, RecursionError):
                return None
            if isinstance(parsed, dict):
                return parsed
            return None
        return None

    async def _read_file(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            content = await self.client.read_file(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        if len(content) > MAX_READ_CHARS:
            content = (
                content[:MAX_READ_CHARS]
                + f"\n[... output truncated ...] ({len(content)} chars total)"
            )
        return content

    async def _write_file(self, path: str, content: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, content)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        return f"wrote {path} ({len(content.encode('utf-8'))} bytes)"

    async def _list_files(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            entries = await asyncio.to_thread(self.client.list_files, resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        # Cap the listing at MAX_LIST_ENTRIES so huge trees cannot flood the
        # context, and tell the model when entries were left out so it knows
        # to list a subdirectory instead of assuming it saw everything.
        total = len(entries)
        rels = []
        base = self.workspace.resolve()
        for e in entries[:MAX_LIST_ENTRIES]:
            try:
                rels.append(str(Path(e).relative_to(base)) if Path(e).is_absolute() else e)
            except ValueError:
                rels.append(str(e))
        text = "\n".join(rels)
        omitted = total - len(rels)
        if omitted > 0:
            note = f"[... {omitted} more entries not shown; list a subdirectory ...]"
            text = f"{text}\n{note}" if text else note
        return text

    async def _run_command(self, command: str) -> str:
        if not command.strip():
            return "error: empty command"
        # The hardened check subsumes the legacy pattern list: it normalizes
        # the command, checks every segment against DENIED_COMMANDS and
        # DENIED_PROGRAMS, and denies force pushes and redirections outside
        # the workspace.
        if is_command_denied(command, self.workspace):
            return "error: command denied"
        try:
            shlex.split(command)
        except ValueError:
            return "error: could not parse command"
        timeout = float(getattr(self.settings, "command_timeout", 300.0) or 300.0)
        # On POSIX the command runs in its own session/process group so that a
        # timeout can kill the whole tree (children and grandchildren included)
        # instead of leaking orphans. On other platforms we fall back to
        # killing the direct child only.
        new_session = os.name == "posix"
        spawn_kwargs: dict[str, Any] = {"start_new_session": True} if new_session else {}
        try:
            if os.name == "posix":
                proc = await asyncio.create_subprocess_exec(
                    "/bin/sh", "-c", command,
                    cwd=str(self.workspace),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
            else:
                parts = shlex.split(command)
                if not parts:
                    return "error: empty command"
                proc = await asyncio.create_subprocess_exec(
                    *parts,
                    cwd=str(self.workspace),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill_process_tree(proc, new_session)
            return "command timed out"
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        combined = out + "\n[stderr]\n" + err if err.strip() else out
        cap = 20000
        if len(combined) > cap:
            combined = "[... output truncated ...]\n" + combined[-cap:]
        if proc.returncode is not None and proc.returncode != 0:
            combined += f"\n[exit status {proc.returncode}]"
        return combined

    @staticmethod
    async def _kill_process_tree(proc: asyncio.subprocess.Process, new_session: bool) -> None:
        """Kill a timed-out command together with every process it spawned."""
        if new_session:
            try:
                pgid = os.getpgid(proc.pid)
            except (OSError, AttributeError):
                pgid = None
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, AttributeError):
                    pass
                else:
                    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    return
        # Fallback: no process group available (or killing it failed).
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()