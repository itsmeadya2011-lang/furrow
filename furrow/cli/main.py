"""Command line entry point for Furrow.

``furrow start`` runs the plan/execute/test loop and exits 0 only when the goal was
actually completed; ``furrow web`` serves the streaming UI.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from furrow.config import Settings, get_settings
from furrow.core.orchestrator import LoopResult, Orchestrator
from furrow.llm import LLMClient, LLMError

console = Console()
error_console = Console(stderr=True)

# provider -> (env var holding the key, Settings field holding the same key)
API_KEYS: dict[str, tuple[str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    "openai": ("OPENAI_API_KEY", "openai_api_key"),
}

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130  # the conventional code for SIGINT

__all__ = ["main", "start", "web"]


@click.group()
def main() -> None:
    """Furrow: an autonomous coding agent that loops until your goal is met."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider to use.",
)
@click.option(
    "--model",
    default=None,
    metavar="MODEL",
    help="Model for every role (planner, worker, tester) unless overridden per role.",
)
@click.option(
    "--planner-model",
    default=None,
    metavar="MODEL",
    help="Model for the planning role.",
)
@click.option(
    "--worker-model",
    default=None,
    metavar="MODEL",
    help="Model for the code-writing role.",
)
@click.option(
    "--tester-model",
    default=None,
    metavar="MODEL",
    help="Model for the testing role.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    metavar="N",
    help="Stop after N cycles (0 means unlimited, the default).",
)
@click.option(
    "--parallel",
    type=int,
    default=None,
    metavar="N",
    help="Maximum number of tasks to run concurrently.",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory Furrow may read and write.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default=None,
    help="Python logging level for the run.",
)
@click.option("--debug", is_flag=True, help="Print a traceback when a run fails unexpectedly.")
def start(
    goal: str | None,
    provider: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    parallel: int | None,
    workspace: Path | None,
    log_level: str | None,
    debug: bool,
) -> None:
    """Run Furrow until GOAL is met."""
    if goal is None:
        goal = click.prompt("Enter your goal for Furrow", default="", show_default=False)
    goal = goal.strip()
    if not goal:
        raise click.UsageError(
            "A non-empty goal is required, e.g. `furrow start \"add a REST API for notes\"`."
        )

    settings = _settings_from_flags(
        provider=provider,
        model=model,
        planner_model=planner_model,
        worker_model=worker_model,
        tester_model=tester_model,
        max_cycles=max_cycles,
        max_parallel_tasks=parallel,
        workspace=workspace,
        log_level=log_level,
    )
    _configure_logging(settings.log_level)
    missing_key = _missing_api_key(settings)
    if missing_key:
        error_console.print(
            f"[red]No API key for provider {_provider_name(settings)}.[/red]\n"
            f"Set {missing_key} in your environment or a .env file, for example:\n"
            f"    export {missing_key}=sk-..."
        )
        sys.exit(EXIT_FAILURE)

    # The goal is user text: escape it so markup in it is printed, not rendered.
    console.print(f"[bold]Furrow[/bold] starting on [bold]{escape(goal)}[/bold]")
    orchestrator = Orchestrator(goal=goal, client=LLMClient(settings=settings))
    try:
        result = asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except LLMError as exc:
        error_console.print(f"[red]LLM request failed:[/red] {escape(str(exc))}")
        _print_api_key_hint(settings)
        sys.exit(EXIT_FAILURE)
    except ValueError as exc:
        # LLMClient raises a plain ValueError when the provider has no API key.
        error_console.print(f"[red]Configuration error:[/red] {escape(str(exc))}")
        _print_api_key_hint(settings)
        sys.exit(EXIT_FAILURE)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary: no traceback unless asked
        error_console.print(f"[red]Furrow failed:[/red] {escape(str(exc))}")
        if debug:
            error_console.print_exception()
        else:
            error_console.print("Re-run with [bold]--debug[/bold] for a full traceback.")
        sys.exit(EXIT_FAILURE)

    _report(result)
    sys.exit(EXIT_OK if result.passed else EXIT_FAILURE)


@main.command()
@click.option(
    "--host",
    default=None,
    metavar="HOST",
    help=(
        "Interface to bind (default: 127.0.0.1). Anything else publishes an "
        "unauthenticated agent that can run commands and write files."
    ),
)
@click.option("--port", default=None, type=int, metavar="PORT", help="TCP port to listen on.")
@click.option(
    "--log-level",
    type=click.Choice(["critical", "error", "warning", "info", "debug", "trace"]),
    default=None,
    help="uvicorn log level.",
)
def web(host: str | None, port: int | None, log_level: str | None) -> None:
    """Serve the Furrow web UI."""
    # Imported here so `furrow start` does not pay for fastapi/uvicorn at startup.
    from furrow.web.server import DEFAULT_HOST, DEFAULT_PORT, run

    bind_host = host or DEFAULT_HOST
    bind_port = port or DEFAULT_PORT
    console.print(f"[bold]Furrow web UI[/bold] on http://{bind_host}:{bind_port}")
    run(host=bind_host, port=bind_port, log_level=log_level)


def _configure_logging(level: str) -> None:
    """Route stdlib logging (LLM retries, warnings) through stderr at the given level.

    ``log_level`` is a free-form string in Settings, so an unusable value from the
    environment falls back to INFO instead of crashing the run.
    """
    numeric = logging.getLevelName(level.strip().upper())
    if not isinstance(numeric, int):
        error_console.print(f"[yellow]Unknown log level {level!r}; using INFO.[/yellow]")
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _settings_from_flags(**flags: Any) -> Settings:
    """Build the run settings from CLI flags on top of the env/.env configuration.

    ``get_settings(**overrides)`` rebuilds and re-caches the settings, and the result
    is handed to ``LLMClient`` explicitly, so the run does not depend on the cached
    singleton staying untouched.
    """
    overrides = {key: value for key, value in flags.items() if value is not None}
    model = overrides.get("model")
    if model:
        # Every call site passes an explicit per-role model, so --model has to fill
        # the three role models too; an explicit --<role>-model flag still wins.
        for role in ("planner_model", "worker_model", "tester_model"):
            overrides.setdefault(role, model)
    return get_settings(**overrides)


def _provider_name(settings: Settings) -> str:
    """Return the provider's plain value; ``str(Provider.X)`` is ``'Provider.X'``."""
    return str(getattr(settings.provider, "value", settings.provider))


def _missing_api_key(settings: Settings) -> str | None:
    """Return the env var name of the API key this run needs but cannot find."""
    entry = API_KEYS.get(_provider_name(settings))
    if entry is None:  # ollama needs no key
        return None
    env, field = entry
    if getattr(settings, field, None) or os.getenv(env):
        return None
    return env


def _print_api_key_hint(settings: Settings) -> None:
    """Tell the user how to give Furrow credentials."""
    missing = _missing_api_key(settings)
    if missing is None:
        error_console.print(
            "Check the provider credentials and network access, then try again."
        )
        return
    error_console.print(f"Set [bold]{missing}[/bold] in your environment or .env file.")


def _report(result: LoopResult) -> None:
    """Print the end-of-run summary."""
    table = Table(title="Furrow run", show_header=False, box=None, pad_edge=False)
    table.add_row("goal", escape(result.goal))
    table.add_row("cycles run", str(result.cycles))
    table.add_row("stopped because", result.stopped_reason)
    table.add_row("tasks completed", str(result.tasks_completed))
    table.add_row("tasks failed", str(result.tasks_failed))
    table.add_row(
        "tests passed",
        "[green]yes[/green]" if result.passed else "[red]no[/red]",
    )
    console.print()
    console.print(table)
    if result.failures:
        console.print("[red]failures:[/red]")
        for failure in result.failures:
            console.print(f"  - {escape(failure)}", highlight=False)
    if result.passed:
        console.print("[bold green]Goal complete.[/bold green]")
    else:
        console.print(f"[bold yellow]Stopped without completing the goal: {result.stopped_reason}.")


if __name__ == "__main__":
    main()  # pragma: no cover - module entry point
