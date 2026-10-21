"""Shared helpers to manage the child processes started by the agents."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def kill_process_tree(proc: asyncio.subprocess.Process, new_session: bool = True) -> None:
    """Kill *proc* together with every process it spawned."""
    if new_session and os.name == "posix":
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
