"""Shared process management helpers for Furrow.

Both the worker and the tester run external commands as subprocesses and need
to kill an entire process tree when a timeout occurs.  This module centralises
that logic so the two call sites stay consistent.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import Any


async def kill_process_tree(
    proc: asyncio.subprocess.Process,
    *,
    new_session: bool = True,
    wait_timeout: float = 5.0,
) -> None:
    """Kill *proc* together with every process it spawned.

    On POSIX the command is started in its own session/process group, so the
    whole group is killed with ``SIGKILL``.  On other platforms only the direct
    child is killed.  Any failure to kill or wait is swallowed: a timeout must
    never mask the original command result.
    """
    if os.name == "posix" and new_session:
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
                    await asyncio.wait_for(proc.wait(), timeout=wait_timeout)
                return
    # Fallback: no process group available (or killing it failed).
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError, OSError):
        await asyncio.wait_for(proc.wait(), timeout=wait_timeout)


def _read_partial_stream(stream: Any) -> bytes:
    """Best-effort read of whatever bytes are buffered on *stream* without blocking.

    Returns ``b""`` when the stream is not a stream-like object or when nothing
    can be read.  Never raises.
    """
    if stream is None:
        return b""
    try:
        return stream.read()
    except Exception:  # pragma: no cover - best effort
        return b""
