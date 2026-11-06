"""Shared process management utilities for Furrow agents."""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def kill_process_tree(
    proc: asyncio.subprocess.Process, new_session: bool | None = None
) -> None:
    """Kill a process and every descendant it spawned.

    On POSIX the command runs in its own session/process group, so the whole
    tree is killed with a single ``killpg``. On other platforms we fall back
    to killing the direct child only.

    After sending the signal the process is waited on (with a short timeout)
    so the caller does not leave a zombie. A failing kill (the process already
    exited, or no process group is available) is ignored and the fallback
    ``proc.kill()`` is used instead.
    """
    if new_session is None:
        new_session = os.name == "posix"

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