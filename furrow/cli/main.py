from __future__ import annotations

import asyncio
import sys
import threading
from queue import Queue

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


def _input_listener(input_queue: Queue[str]) -> None:
    while True:
        try:
            line = input()
            input_queue.put(line)
        except (EOFError, KeyboardInterrupt):
            break


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Max cycles before halting")
def start(goal: str | None, model: str | None, max_cycles: int | None) -> None:
    from furrow.config import settings
    configure_logging(level=settings.log_level)
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    input_queue: Queue[str] = Queue()
    listener_thread = threading.Thread(target=_input_listener, args=(input_queue,), daemon=True)
    listener_thread.start()
    orchestrator = Orchestrator(goal=goal)
    if orchestrator.load_state():
        console.print("[yellow]Resumed from previous state.[/yellow]")
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
