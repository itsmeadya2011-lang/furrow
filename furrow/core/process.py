"""Shared process-management utilities for Furrow.

Both the worker and the tester need to kill a timed-out command together with
every process it spawned, so this logic lives here in a single place instead of
being duplicated.
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

    On POSIX the command runs in its own session/process group (when
    ``new_session`` is true), so a timeout can kill the whole tree instead of
    leaking orphans.  On other platforms we fall back to killing the direct
    child only.
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
