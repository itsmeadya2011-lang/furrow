from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
import structlog
from rich.console import Console
from rich.panel import Panel

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


def setup_logging(level: str = "INFO") -> None:
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
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper()))


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    setup_logging(log_level)
    settings.log_level = log_level


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--resume", is_flag=True, help="Resume from previous session")
@click.option("--session-path", default=None, help="Path to session file")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    resume: bool,
    session_path: str | None,
) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles

    session_file = Path(session_path) if session_path else Path.cwd() / ".furrow_session.json"

    if resume and not session_file.exists():
        console.print("[red]No session file found to resume.[/red]")
        sys.exit(1)

    try:
        asyncio.run(
            Orchestrator(
                goal=goal or "",
                max_cycles=max_cycles or settings.max_cycles,
                session_path=session_file if resume else None,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


@main.command()
def usage() -> None:
    """Show token usage statistics."""
    client = LLMClient()
    usage = client.get_token_usage()
    console.print(f"[bold]Token Usage:[/bold]")
    console.print(f"  Input:  {usage['input']:,}")
    console.print(f"  Output: {usage['output']:,}")
    console.print(f"  Total:  {usage['total']:,}")


@main.command()
@click.argument("session_file", type=click.Path(exists=True))
def inspect(session_file: str) -> None:
    """Inspect a session file."""
    import json
    from rich.pretty import Pretty

    data = json.loads(Path(session_file).read_text())
    console.print(Panel(Pretty(data), title=f"Session: {session_file}"))


if __name__ == "__main__":
    main()
