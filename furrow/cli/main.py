from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.prompt import Prompt

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--interactive", "-i", is_flag=True, help="Prompt for input after each cycle")
def start(goal: str | None, model: str | None, interactive: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    
    async def on_cycle_end() -> str | None:
        if interactive:
            console.print("\n[dim]Press Enter to continue, or type a new goal to update...[/dim]")
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, Prompt.ask, "New goal (or Enter to continue)", default="", show_default=False
                )
                if user_input.strip():
                    return user_input.strip()
            except (EOFError, KeyboardInterrupt):
                return "STOP"
        return None
    
    try:
        orchestrator = Orchestrator(goal=goal, on_cycle_end=on_cycle_end if interactive else None)
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
