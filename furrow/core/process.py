"""Shared process-management helpers for Furrow.

Both the worker and the tester run subprocesses that must be killed together
with every process they spawned when a timeout fires.  This module provides a
single implementation so the two call sites never drift apart.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def kill_process_tree(
    proc: asyncio.subprocess.Process,
    *,
    new_session: bool = True,
    kill_timeout: float = 5.0,
) -> None:
    """Kill *proc* together with every process it spawned.

    On POSIX the command runs in its own session/process group so that a
    timeout can kill the whole tree (children and grandchildren included)
    instead of leaking orphans.  On other platforms we fall back to killing
    the direct child only.

    Parameters
    ----------
    proc:
        The subprocess to kill.
    new_session:
        Whether *proc* was started with ``start_new_session=True`` (POSIX only).
    kill_timeout:
        Seconds to wait for the process to exit after sending SIGKILL.
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
                    await asyncio.wait_for(proc.wait(), timeout=kill_timeout)
                return
    # Fallback: no process group available (or killing it failed).
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
