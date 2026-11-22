"""Shared process management utilities for Furrow.

Provides process-tree termination used by both the worker (command execution)
and the tester (test suite execution). Keeping this in one place avoids
duplication and ensures consistent behaviour across the codebase.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def kill_process_tree(
    proc: asyncio.subprocess.Process, new_session: bool = True
) -> None:
    """Kill *proc* together with every process it spawned.

    On POSIX the command runs in its own session/process group, so a timeout
    can kill the whole tree (children and grandchildren included) instead of
    leaking orphans. On other platforms we fall back to killing the direct
    child only.

    Never raises.
    """
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
    with contextlib.suppress(Exception):
        await proc.wait()
