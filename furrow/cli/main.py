from __future__ import annotations

import asyncio
import inspect
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.core.orchestrator import Orchestrator

console = Console()


def _render_event(event: dict[str, Any]) -> None:
    etype = event.get("type", "")
    if etype == "plan":
        console.print(Panel(Pretty(event.get("data", event)), title="Plan", border_style="blue"))
    elif etype == "task_started":
        tid = event.get("task_id", "?")
        console.print(f"[cyan]▶ Task {tid} started[/cyan]")
    elif etype == "task_completed":
        tid = event.get("task_id", "?")
        console.print(f"[green]✓ Task {tid} completed[/green]")
    elif etype == "task_failed":
        tid = event.get("task_id", "?")
        err = event.get("error", "")
        console.print(f"[red]✗ Task {tid} failed: {err}[/red]")
    elif etype == "test_result":
        passed = event.get("passed", False)
        summary = event.get("summary", "")
        if passed:
            console.print(f"[green]Tests passed: {summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {summary}[/red]")
            for failure in event.get("failures", []):
                console.print(f"  • [red]{failure}[/red]")
    elif etype == "cycle_start":
        cycle = event.get("cycle", event.get("data", ""))
        console.print(f"\n[bold cyan]═══ Cycle {cycle} ═══[/bold cyan]")
    elif etype == "cycle_end":
        console.print(f"[dim]─── end cycle ───[/dim]")
    elif etype == "done":
        status = event.get("status", "complete")
        border = "green" if status in ("complete", "success") else "yellow"
        console.print(Panel(Pretty(event.get("data", event)), title="Done", border_style=border))
    else:
        console.print(Pretty(event))


async def _drive(orchestrator: Orchestrator) -> None:
    if hasattr(orchestrator, "on_event"):
        attr = orchestrator.on_event

        async def _on_event_async(event: dict[str, Any]) -> None:
            _render_event(event)

        def _on_event_sync(event: dict[str, Any]) -> None:
            _render_event(event)

        if callable(attr) and not inspect.iscoroutinefunction(attr):
            orchestrator.on_event = _on_event_sync
        else:
            orchestrator.on_event = _on_event_async
        await orchestrator.run()
        return

    if hasattr(orchestrator, "stream"):
        stream = orchestrator.stream()
        if hasattr(stream, "__aiter__"):
            async for event in stream:
                _render_event(event)
            return

        async def _aiter_from_sync() -> Any:
            for event in stream:
                yield event

        async for event in _aiter_from_sync():
            _render_event(event)
        return

    await orchestrator.run()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    orchestrator = Orchestrator(goal=goal)
    try:
        asyncio.run(_drive(orchestrator))
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()