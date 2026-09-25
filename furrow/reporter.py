from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from typing import Any, Protocol

from rich.console import Console


class Reporter(Protocol):
    """Output sink for orchestrator progress messages."""

    def print(self, message: Any) -> None: ...


class ConsoleReporter:
    """Render progress messages to a rich console (CLI default)."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print(self, message: Any) -> None:
        self.console.print(message)


class QueueReporter:
    """Push plain-text renderings of messages onto an asyncio.Queue.

    Rich markup / renderables (e.g. Panel) are rendered with no colour system so
    that websocket consumers receive readable, colour-free output.
    """

    def __init__(self, queue: asyncio.Queue[str]) -> None:
        self.queue = queue

    def print(self, message: Any) -> None:
        buf = io.StringIO()
        console = Console(
            file=buf, width=120, color_system=None, force_terminal=False, highlight=False
        )
        console.print(message)
        text = buf.getvalue().rstrip()
        if text:
            self.queue.put_nowait(text)


async def drain_queue(queue: asyncio.Queue[str], send: Callable[[str], Any]) -> None:
    """Forward every message from *queue* to *send* until the sender errors out."""
    while True:
        message = await queue.get()
        try:
            await send(message)
        except Exception:  # noqa: BLE001 - any send failure stops the stream
            break
