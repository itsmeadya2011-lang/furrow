from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
def version() -> None:
    """Print the Furrow version."""
    click.echo("furrow 0.1.0")


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider",
)
@click.option("--max-cycles", type=int, default=None, help="Stop after N cycles")
@click.option("--parallel", type=int, default=None, help="Max parallel tasks")
@click.option("--workspace", default=None, help="Workspace directory")
@click.option("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ...)")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    parallel: int | None,
    workspace: str | None,
    log_level: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if parallel is not None:
        settings.max_parallel_tasks = parallel
    if workspace is not None:
        settings.workspace = Path(workspace)
    if log_level is not None:
        settings.log_level = log_level
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def status() -> None:
    """Show the persisted loop state."""
    state_file = Path(settings.state_file)
    if not state_file.exists():
        console.print("No state found. Run `furrow start <goal>` first.")
        return
    try:
        raw = state_file.read_text()
        data = json.loads(raw)
    except Exception as exc:
        console.print(f"[red]Could not read state file {state_file}: {exc}[/red]")
        raise SystemExit(1) from exc
    console.print_json(json.dumps(data, indent=2))


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
