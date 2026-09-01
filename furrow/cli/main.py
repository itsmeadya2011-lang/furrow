from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from furrow.core.orchestrator import Orchestrator
from furrow.core.session import (
    SessionCorruptedError,
    SessionManager,
    SessionNotFoundError,
)

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--resume",
    "session_id",
    default=None,
    help="Resume a previously saved session by ID.",
)
@click.option(
    "--workspace",
    "workspace",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Workspace directory (defaults to settings).",
)
def start(goal: str | None, model: str | None, session_id: str | None, workspace: Path | None) -> None:
    if model:
        from furrow.config import settings

        settings.model = model
    if workspace is not None:
        from furrow.config import settings

        settings.workspace = workspace

    try:
        if session_id:
            try:
                orchestrator = Orchestrator.from_session(session_id=session_id)
            except SessionNotFoundError as exc:
                console.print(f"[red]Session not found:[/red] {session_id}")
                console.print(f"  {exc}")
                sys.exit(2)
            except SessionCorruptedError as exc:
                console.print(f"[red]Session file is corrupted:[/red] {session_id}")
                console.print(f"  {exc}")
                sys.exit(2)
        else:
            if not goal:
                goal = click.prompt("Enter your goal for Furrow")
            from furrow.config import settings

            manager = SessionManager(settings.workspace)
            new_id, _state = manager.new_session(goal=goal, workspace=settings.workspace)
            console.print(f"[cyan]New session:[/cyan] {new_id}")
            orchestrator = Orchestrator(goal=goal, session_id=new_id, session_manager=manager)

        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command(name="list-sessions")
@click.option(
    "--workspace",
    "workspace",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Workspace directory (defaults to settings).",
)
def list_sessions(workspace: Path | None) -> None:
    from furrow.config import settings

    ws = workspace or settings.workspace
    manager = SessionManager(ws)
    sessions = manager.list_sessions()
    if not sessions:
        console.print(f"[dim]No saved sessions in {manager.sessions_dir}[/dim]")
        return

    table = Table(title=f"Sessions in {manager.sessions_dir}", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Cycles", justify="right")
    table.add_column("Goal")
    table.add_column("Updated", style="dim")

    for s in sessions:
        status_color = {
            "running": "yellow",
            "paused": "blue",
            "completed": "green",
        }.get(s.status, "white")
        goal_preview = s.goal if len(s.goal) <= 60 else s.goal[:57] + "..."
        table.add_row(
            s.session_id,
            f"[{status_color}]{s.status}[/{status_color}]",
            str(s.cycles),
            goal_preview,
            s.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
    console.print(table)


@main.command(name="delete-session")
@click.argument("session_id")
@click.option(
    "--workspace",
    "workspace",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Workspace directory (defaults to settings).",
)
def delete_session(session_id: str, workspace: Path | None) -> None:
    from furrow.config import settings

    ws = workspace or settings.workspace
    manager = SessionManager(ws)
    if manager.delete(session_id):
        console.print(f"[green]Deleted session:[/green] {session_id}")
    else:
        console.print(f"[red]Session not found:[/red] {session_id}")
        sys.exit(2)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
