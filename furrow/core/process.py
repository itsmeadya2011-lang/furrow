"""Process management utilities for Furrow.

A single, shared implementation of process-group killing so that the worker
and the tester do not each maintain their own (slightly different) version.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def kill_process_tree(proc: asyncio.subprocess.Process, new_session: bool = True) -> None:
    """Kill *proc* together with every process it spawned.

    On POSIX, kills the entire process group when *new_session* is True (the
    default), so children and grandchildren are cleaned up instead of being
    left as orphans.  Falls back to killing the direct child only when no
    process group is available.  Never raises.
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
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()