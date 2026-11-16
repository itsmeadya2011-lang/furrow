"""Command execution utilities for the worker agent."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from pathlib import Path
from typing import Any

from furrow.agents.security import is_command_denied
from furrow.config import Settings


async def run_command(command: str, workspace: Path, settings: Settings) -> str:
    """Run a shell command in the workspace and return its output."""
    if not command.strip():
        return "error: empty command"
    # The hardened check subsumes the legacy pattern list: it normalizes
    # the command, checks every segment against DENIED_COMMANDS and
    # DENIED_PROGRAMS, and denies force pushes and redirections outside
    # the workspace.
    if is_command_denied(command, workspace):
        return "error: command denied"
    try:
        shlex.split(command)
    except ValueError:
        return "error: could not parse command"
    timeout = float(getattr(settings, "command_timeout", 300.0) or 300.0)
    # On POSIX the command runs in its own session/process group so that a
    # timeout can kill the whole tree (children and grandchildren included)
    # instead of leaking orphans. On other platforms we fall back to
    # killing the direct child only.
    new_session = os.name == "posix"
    spawn_kwargs: dict[str, Any] = {"start_new_session": True} if new_session else {}
    try:
        if os.name == "posix":
            proc = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-c",
                command,
                cwd=str(workspace),
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
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **spawn_kwargs,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        return f"error: {exc}"
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _kill_process_tree(proc, new_session)
        return "command timed out"
    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")
    combined = out + "\n[stderr]\n" + err if err.strip() else out
    cap = 20000
    if len(combined) > cap:
        combined = "[... output truncated ...]\n" + combined[-cap:]
    return combined


async def _kill_process_tree(
    proc: asyncio.subprocess.Process, new_session: bool
) -> None:
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