"""Shared process-management helpers for Furrow.

Both the worker and the tester spawn subprocesses and need to kill an entire
process tree when a command times out.  This module centralises that logic so
it is implemented once and reused everywhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import Any


async def kill_process_tree(
    proc: asyncio.subprocess.Process, new_session: bool
) -> None:
    """Kill *proc* together with every process it spawned.

    On POSIX the command runs in its own session/process group so that a
    timeout kills the whole tree (children and grandchildren included) instead
    of leaking orphans.  On other platforms we fall back to killing the direct
    child only.

    Never raises.
    """
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
    with contextlib.suppress(Exception):
        await proc.wait()
