"""Command line entry points: ``furrow start`` and ``furrow web``."""

from __future__ import annotations

import asyncio
import importlib.metadata
import sys
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130

# ``stopped_reason`` values that still count as a successful run.
_SUCCESS_REASONS = frozenset(
    {
        "goal_complete",
        "goal complete",
        "goal_completed",
        "complete",
        "completed",
        "done",
        "success",
        "finished",
    }
)


def _resolve_version() -> str:
    """Distribution version, with a safe fallback when furrow is not installed."""
    try:
        return importlib.metadata.version("furrow")
    except importlib.metadata.PackageNotFoundError:
        module = sys.modules.get("furrow")
        attr = getattr(module, "__version__", None)
        return str(attr) if attr else "0.0.0+unknown"


def _run_failed(result: Any) -> bool:
    """True when the finished run should exit non-zero."""
    if not isinstance(result, dict):
        return True
    reason = result.get("stopped_reason")
    if reason and str(reason).strip().lower() not in _SUCCESS_REASONS:
        return True
    tasks = result.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, dict) and str(task.get("status")) == "failed":
                return True
    for key in ("failed", "unfinished"):
        value = result.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False


def _summarize(result: Any) -> str:
    if not isinstance(result, dict):
        return "run finished."
    reason = result.get("stopped_reason") or "no reason reported"
    parts = [f"reason: {reason}"]
    for key in ("cycles", "completed", "failed", "unfinished"):
        if key in result:
            parts.append(f"{key}: {result[key]}")
    return " | ".join(parts)


@click.group()
@click.version_option(_resolve_version(), "--version", prog_name="furrow")
def main() -> None:
    """Furrow: an autonomous coding agent loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the LLM model.")
@click.option(
    "--provider",
    default=None,
    type=click.Choice([member.value for member in Provider]),
    help="Override the LLM provider.",
)
@click.option("--max-cycles", default=None, type=int, help="Maximum cycles (0 = unlimited).")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks.")
@click.version_option(_resolve_version(), "--version", prog_name="furrow")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
) -> None:
    """Run Furrow on GOAL."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    updates: dict[str, Any] = {}
    if model:
        updates["model"] = model
    if provider:
        updates["provider"] = Provider(provider)
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if max_parallel is not None:
        updates["max_parallel_tasks"] = max_parallel
    # Never mutate the shared settings singleton; give this run its own copy.
    run_settings = settings.model_copy(update=updates)
    client = LLMClient(settings=run_settings)
    orchestrator = Orchestrator(goal=goal, client=client)
    console.print(f"[bold]Goal:[/bold] {goal}")
    try:
        result = asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    console.print(_summarize(result))
    if _run_failed(result):
        console.print("[red]Furrow run did not complete successfully.[/red]")
        sys.exit(EXIT_FAILURE)
    sys.exit(EXIT_OK)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Interface to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to bind.")
@click.option("--log-level", default=None, help="Uvicorn log level (e.g. debug).")
def web(host: str, port: int, log_level: str | None) -> None:
    """Serve the web UI."""
    from furrow.web.server import run

    console.print(f"[green]Furrow web UI:[/green] http://{host}:{port}")
    run(host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()