"""Shared helpers for managing subprocess lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

logger = logging.getLogger(__name__)


async def kill_process_tree(
    proc: asyncio.subprocess.Process, new_session: bool | None = None
) -> None:
    """Kill ``proc`` together with every process it spawned.

    On POSIX systems, when ``new_session`` is true (the default), the whole
    process group is killed via :func:`os.killpg`.  If that fails for any
    reason the caller falls back to a direct :meth:`Process.kill`.  On
    non-POSIX platforms only the direct kill is attempted.

    After the kill attempt the process is awaited so that callers do not
    leak zombie entries.  All expected error conditions are suppressed so
    the helper never raises.
    """
    if new_session is None:
        new_session = os.name == "posix"

    if new_session and os.name == "posix":
        try:
            pgid = os.getpgid(proc.pid)
        except (OSError, AttributeError):
            pgid = None

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, AttributeError):
                logger.warning(
                    "Falling back to direct process kill for pid %s",
                    getattr(proc, "pid", None),
                )
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
            else:
                with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                return

    # Fallback: no process group available, killing it failed, or non-POSIX.
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()

    with contextlib.suppress(Exception):
        await proc.wait()