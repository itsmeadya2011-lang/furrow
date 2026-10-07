from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--session-file", default=None, help="Path to session file for persistence")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--no-auto-commit", is_flag=True, help="Disable auto-commit to git")
@click.option("--compact-threshold", default=10, help="Number of cycles before compacting context")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    session_file: str | None,
    max_cycles: int,
    no_auto_commit: bool,
    compact_threshold: int,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if max_cycles > 0:
        settings.max_cycles = max_cycles

    session_path = Path(session_file) if session_file else None

    try:
        asyncio.run(Orchestrator(
            goal=goal,
            session_file=session_path,
            auto_commit=not no_auto_commit,
            compact_threshold=compact_threshold,
        ).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
@click.argument("session_file", type=click.Path(exists=True))
def status(session_file: str) -> None:
    """Show the status of a saved session."""
    import json
    from furrow.config import TaskModel

    data = json.loads(Path(session_file).read_text())
    tasks = [TaskModel(**t) for t in data.get("tasks", [])]

    console.print(f"[bold]Session:[/bold] {session_file}")
    console.print(f"[bold]Cycles:[/bold] {data.get('cycles', 0)}")
    console.print(f"[bold]Goal:[/bold] {data.get('goal', 'N/A')}")
    console.print(f"[bold]Original Goal:[/bold] {data.get('original_goal', 'N/A')}")
    console.print(f"[bold]Tasks:[/bold] {len(tasks)}")
    for t in tasks:
        status_color = "green" if t.status == "completed" else ("red" if t.status == "failed" else "yellow")
        console.print(f"  [{status_color}]{t.id}[/{status_color}]: {t.description} ({t.status})")


if __name__ == "__main__":
    main()