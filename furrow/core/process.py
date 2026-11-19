from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def kill_process_tree(
    proc: asyncio.subprocess.Process,
    *,
    new_session: bool,
) -> None:
    """Kill *proc* together with every process it spawned.

    When ``new_session`` is true (POSIX), the whole process group is killed so
    children and grandchildren do not leak as orphans. Otherwise fall back to
    killing the direct child only.
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
