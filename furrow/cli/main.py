from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
def main() -> None:
    """Autonomous coding agent with an infinite parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    """Run the orchestrator against GOAL until it is satisfied."""
    # Imported inside the command so `furrow --help` stays cheap: pulling in the
    # orchestrator at module scope drags the Anthropic and OpenAI SDKs into every
    # CLI invocation, including `furrow web`.
    import asyncio
    import sys

    from furrow.config import settings
    from furrow.core.orchestrator import Orchestrator

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    # `settings` is a process-wide singleton, so copy it instead of mutating it and
    # hand the copy to the orchestrator (which owns it for this run only).
    config = settings.model_copy(update={"model": model}) if model else settings
    try:
        asyncio.run(Orchestrator(goal=goal, settings=config).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        # Exit 130 (128 + SIGINT), the conventional shell code for Ctrl-C. Exiting 0
        # would tell CI and shell scripts that an aborted run succeeded.
        sys.exit(130)


@main.command()
def web() -> None:
    """Serve the web UI (FURROW_WEB_HOST / FURROW_WEB_PORT to override)."""
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
