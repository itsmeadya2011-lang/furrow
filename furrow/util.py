"""Shared utilities for Furrow.

This module collects helpers that are used by more than one layer of the
application (the CLI and the web server in particular) so that their logic is
defined once and imported everywhere.  Keeping them here avoids the verbatim
duplications that otherwise creep into the two entry points.

The module is intentionally dependency-free at import time: it only needs the
standard library and :mod:`typing`, so it can be imported from anywhere without
creating a circular import.
"""

from __future__ import annotations

import inspect
from typing import Any

#: Maximum number of bytes a tool output (command output, file content, test
#: runner output) may contribute to the context before it is truncated.  The
#: value is shared by the worker, the tester and the output parser so that the
#: behaviour is consistent everywhere.
MAX_OUTPUT_BYTES = 20000

#: How long (in seconds) a timed-out process is allowed to linger while its
#: process group is being killed before the orchestrator gives up waiting.
_KILL_TIMEOUT = 5.0


async def release_client(client: Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result is
    awaitable.  A failing close is logged and ignored: closing must never turn
    a successful run into a failed one.
    """
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:  # pragma: no cover - defensive
        # Importing the logger lazily keeps this module free of a logging
        # dependency at import time and avoids a circular import with
        # ``furrow.logging``.
        import logging

        logging.getLogger(__name__).warning("Could not close the LLM client: %s", exc)


async def run_orchestrator(orchestrator: Any) -> Any:
    """Run an orchestrator and release its LLM client afterwards.

    The client is closed in a ``finally`` block, so its HTTP connections are
    released when the run finishes, when it fails and when it is cancelled
    (e.g. because the websocket client went away).
    """
    try:
        return await orchestrator.run()
    finally:
        await release_client(getattr(orchestrator, "client", None))


def truncate_output(text: str, cap: int = MAX_OUTPUT_BYTES) -> str:
    """Return *text* truncated to at most *cap* bytes.

    When the text is too long the *tail* is kept (the end of a command's
    output, the summary at the bottom of a test run, ...) and a marker is
    prepended so the reader knows something was dropped.
    """
    if not text:
        return text
    if len(text) <= cap:
        return text
    return "[... output truncated ...]\n" + text[-cap:]


def truncate_head(text: str, cap: int = MAX_OUTPUT_BYTES) -> str:
    """Return *text* truncated to at most *cap* bytes, keeping the head.

    The head is kept for content where the beginning matters most (a file that
    was read, for instance) and a marker is appended to show truncation.
    """
    if not text:
        return text
    if len(text) <= cap:
        return text
    return text[:cap] + "\n[... output truncated ...]"


async def kill_process_tree(proc: Any, new_session: bool = True) -> None:
    """Kill *proc* together with every process it spawned.

    On POSIX the process is started in its own session so that a timeout can
    kill the whole tree (children and grandchildren included) instead of
    leaking orphans.  On other platforms we fall back to killing the direct
    child only.  The function never raises: a missing process is not an error.
    """
    import asyncio
    import contextlib
    import os
    import signal

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
                    await asyncio.wait_for(proc.wait(), timeout=_KILL_TIMEOUT)
                return
    # Fallback: no process group available (or killing it failed).
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
    with contextlib.suppress(Exception):
        await proc.wait()


__all__ = (
    "MAX_OUTPUT_BYTES",
    "release_client",
    "run_orchestrator",
    "truncate_output",
    "truncate_head",
    "kill_process_tree",
)