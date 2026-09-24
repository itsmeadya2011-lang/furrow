from __future__ import annotations

import asyncio
import logging
import sys

import click
import structlog
from rich.console import Console

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with the given log level."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    configure_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--resume", is_flag=True, help="Resume from saved state")
def start(goal: str | None, model: str | None, max_cycles: int, resume: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    
    state_file = settings.workspace / ".furrow_state.json"
    if not resume and state_file.exists():
        if not click.confirm("State file exists. Overwrite?"):
            console.print("[yellow]Use --resume to continue from saved state.[/yellow]")
            sys.exit(0)
    
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
def init() -> None:
    """Initialize a new Furrow project with default configuration."""
    env_file = settings.workspace / ".env"
    if env_file.exists():
        console.print("[yellow].env already exists[/yellow]")
    else:
        env_file.write_text("# Furrow configuration\n# ANTHROPIC_API_KEY=your_key_here\n# OPENAI_API_KEY=your_key_here\n")
        console.print(f"[green]Created {env_file}[/green]")
    
    state_file = settings.workspace / ".furrow_state.json"
    if state_file.exists():
        console.print("[yellow]State file already exists[/yellow]")
    else:
        console.print("[green]Ready to start. Run 'furrow start <goal>'[/green]")


if __name__ == "__main__":
    main()