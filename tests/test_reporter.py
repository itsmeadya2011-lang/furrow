from __future__ import annotations

import asyncio
import re

from rich.console import Console

from furrow.reporter import ConsoleReporter, QueueReporter, drain_queue


def test_console_reporter_outputs_text(capsys):
    console = Console(force_terminal=False, width=80)
    reporter = ConsoleReporter(console=console)
    reporter.print("[bold green]hello world[/bold green]")
    captured = capsys.readouterr().out
    assert "hello world" in captured


async def test_queue_reporter_strips_markup():
    q: asyncio.Queue[str] = asyncio.Queue()
    reporter = QueueReporter(q)
    reporter.print("[red]error[/red] message")
    msg = q.get_nowait()
    assert "error" in msg
    assert "message" in msg
    assert re.search(r"\[/red\]|\[red\]", msg) is None


async def test_queue_reporter_renders_panel():
    from rich.panel import Panel

    q: asyncio.Queue[str] = asyncio.Queue()
    reporter = QueueReporter(q)
    reporter.print(Panel("hello panel", title="Plan"))
    msg = q.get_nowait()
    assert "hello panel" in msg
    assert "Plan" in msg


async def test_queue_reporter_ignores_empty():
    q: asyncio.Queue[str] = asyncio.Queue()
    reporter = QueueReporter(q)
    reporter.print("")
    assert q.empty()


async def test_drain_queue_forwards_messages():
    q: asyncio.Queue[str] = asyncio.Queue()
    sent: list[str] = []

    async def send(message: str):
        sent.append(message)

    q.put_nowait("a")
    q.put_nowait("b")
    task = asyncio.create_task(drain_queue(q, send))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert sent == ["a", "b"]


async def test_drain_queue_stops_on_send_error():
    q: asyncio.Queue[str] = asyncio.Queue()
    q.put_nowait("a")

    async def send(message: str):
        raise RuntimeError("ws closed")

    task = asyncio.create_task(drain_queue(q, send))
    await asyncio.sleep(0.05)
    assert task.done()
