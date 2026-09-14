from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--state-file", default=None, help="Path to state file for persistence")
def start(goal: str | None, model: str | None, max_cycles: int, state_file: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles

    state_path = Path(state_file) if state_file else None
    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_path).run())
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
def init() -> None:
    """Initialize a new Furrow project with example configuration."""
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text("""# Furrow Configuration
# FURROW_PROVIDER=anthropic
# FURROW_MODEL=claude-sonnet-4-20250514
# FURROW_ANTHROPIC_API_KEY=your-api-key-here
# FURROW_MAX_CYCLES=10
# FURROW_LOG_LEVEL=INFO
""")
        console.print("[green]Created .env file[/green]")
    else:
        console.print("[yellow].env already exists[/yellow]")
    
    state_file = Path(".furrow_state.json")
    if not state_file.exists():
        state_file.write_text("{}")
        console.print("[green]Created .furrow_state.json[/green]")


if __name__ == "__main__":
    main()
