"""Command-line entry point for the ``furrow`` executable.

The CLI stays deliberately thin. It parses and validates options, folds them
into the single :class:`~furrow.config.Settings` singleton resolved by
:func:`furrow.config.get_settings`, drives
:class:`~furrow.core.orchestrator.Orchestrator`, and maps the outcome onto a
process exit code.

Exit codes
----------
``0``
    The run reported ``RunResult.completed``.
``1``
    The run finished but did not complete the goal (failing tasks or tests,
    an exhausted cycle budget, or an unhandled error).
``2``
    Configuration was rejected before any work started.
``130``
    The operator interrupted the run with ``Ctrl-C``.

Import policy
-------------
``click``, ``rich`` and :mod:`furrow.config` are cheap and imported at module
scope. Everything that drags in the agent implementations
(:mod:`furrow.core.orchestrator`) or an HTTP stack
(:mod:`furrow.web.server`) is imported inside the command that needs it, so
``furrow --help`` and ``furrow --version`` stay instant, work without an API
client being importable, and never require fastapi or uvicorn to be installed.
``Orchestrator`` is also constructed with the resolved settings instance rather
than the module-level singleton so the CLI cannot be broken by another
module's import order.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from furrow import __version__
from furrow.config import LOG_LEVELS, Provider, Settings, configure_logging, get_settings

if TYPE_CHECKING:  # Runtime import would defeat the deferred-import policy above.
    from furrow.core.events import FurrowEvent, RunResult

console = Console()

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130

_PROVIDER_CHOICES = click.Choice([p.value for p in Provider], case_sensitive=False)

_LOG_LEVEL_CHOICES = click.Choice(list(LOG_LEVELS), case_sensitive=False)


class _NullStatus:
    """No-op stand-in for :class:`rich.status.Status` used under ``--quiet``."""

    def __enter__(self) -> None:
        """Enter the (absent) spinner."""

    def __exit__(self, *exc_info: object) -> None:
        """Leave the (absent) spinner; never swallow exceptions."""


def _abort(message: str) -> None:
    """Print a configuration error and exit with :data:`EXIT_CONFIG`.

    Args:
        message: Actionable text describing what is wrong and how to fix it.
    """
    console.print(f"[bold red]error:[/bold red] {message}")
    raise SystemExit(EXIT_CONFIG)


def _validate_options(
    max_parallel: int | None, max_cycles: int | None, workspace: str | None
) -> None:
    """Reject nonsensical option values before any work or API call happens.

    Args:
        max_parallel: Value of ``--max-parallel``, or ``None`` when unset.
        max_cycles: Value of ``--max-cycles``, or ``None`` when unset.
        workspace: Value of ``--workspace``, or ``None`` when unset.

    Raises:
        click.BadParameter: If any provided value is out of range or the
            workspace path is not an existing directory.
    """
    if max_parallel is not None and max_parallel < 1:
        raise click.BadParameter("must be 1 or greater", param_hint="--max-parallel")
    if max_cycles is not None and max_cycles < 0:
        raise click.BadParameter(
            "must be 0 (unlimited) or greater", param_hint="--max-cycles"
        )
    if workspace is not None and not Path(workspace).expanduser().is_dir():
        raise click.BadParameter("is not a directory", param_hint="--workspace")


def _configure(
    settings: Settings,
    provider: str | None,
    model: str | None,
    workspace: str | None,
    max_parallel: int | None,
    max_cycles: int | None,
    test_command: str | None,
) -> tuple[Provider, str]:
    """Fold CLI options into the settings singleton and check provider credentials.

    ``--model`` sets ``settings.override_model`` rather than ``settings.model``:
    every agent passes its own role-specific model to ``LLMClient.complete``,
    which resolves ``override_model or model or settings.model``. Writing to
    ``settings.model`` therefore had no effect and the flag was a silent no-op.

    Args:
        settings: The shared settings instance to mutate.
        provider: Raw ``--provider`` value, or ``None`` to keep the configured one.
        model: Raw ``--model`` value, or ``None`` to keep the configured models.
        workspace: Raw ``--workspace`` value, or ``None`` to keep the default.
        max_parallel: Raw ``--max-parallel`` value, or ``None``.
        max_cycles: Raw ``--max-cycles`` value, or ``None``.
        test_command: Raw ``--test-command`` value, or ``None``.

    Returns:
        The effective provider and the effective model name.

    Raises:
        SystemExit: With :data:`EXIT_CONFIG` when the provider needs an API key
            and none can be resolved.
    """
    effective_provider = settings.provider
    if provider is not None:
        effective_provider = Provider(provider.lower())
        settings.provider = effective_provider
    if model is not None:
        settings.override_model = model
    if workspace is not None:
        settings.workspace = Path(workspace).expanduser().resolve()
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if test_command is not None:
        settings.test_command = test_command

    needs_key = effective_provider is not Provider.OLLAMA
    if needs_key and settings.api_key_for(effective_provider) is None:
        variable = f"{effective_provider.value.upper()}_API_KEY"
        _abort(
            f"no API key resolved for provider '{effective_provider.value}'. "
            f"Export {variable}, put it in .env as FURROW_{variable}, or run "
            f"with --provider ollama."
        )
    effective_model = settings.override_model or settings.model
    return effective_provider, effective_model


def _make_reporter() -> Callable[[FurrowEvent], None]:
    """Build the synchronous ``on_event`` callback used for terminal output.

    The callback runs on the event loop, possibly from inside async code, so it
    only formats and prints; it never awaits.

    Returns:
        A callable that renders one :class:`~furrow.core.events.FurrowEvent`.
    """
    from furrow.core.events import EventType

    styles = {
        EventType.RUN_START: "bold green",
        EventType.CYCLE_START: "bold cyan",
        EventType.PLAN: "cyan",
        EventType.TASK_START: "blue",
        EventType.TASK_COMPLETE: "green",
        EventType.TASK_FAILED: "red",
        EventType.TASK_SKIPPED: "yellow",
        EventType.TEST_START: "magenta",
        EventType.TEST_RESULT: "magenta",
        EventType.CYCLE_END: "bold cyan",
        EventType.ERROR: "red",
    }
    suppressed = frozenset({EventType.PLAN, EventType.CYCLE_END, EventType.TEST_RESULT})

    def report(event: FurrowEvent) -> None:
        """Print `event` unless it is high-volume noise already in the summary."""
        if event.type in suppressed:
            return
        prefix = f"cycle {event.cycle}: " if event.cycle else ""
        label = f"{event.type.value} {event.task_id}".strip()
        text = event.message or label
        style = styles.get(event.type, "white")
        console.print(f"[{style}]{prefix}{label}[/] {text}", highlight=False)

    return report


def _render_banner(
    goal: str, provider: Provider, model: str, settings: Settings, quiet: bool
) -> None:
    """Print the run header describing the resolved configuration.

    Args:
        goal: The goal being pursued.
        provider: Effective provider name.
        model: Effective model name.
        settings: The resolved settings, for workspace and budget values.
        quiet: Suppress the banner when set.
    """
    if quiet:
        return
    cycles = settings.max_cycles or "unlimited"
    body = "\n".join(
        [
            f"[bold]Goal:[/bold] {goal}",
            f"[bold]Provider:[/bold] {provider.value}    [bold]Model:[/bold] {model}",
            f"[bold]Workspace:[/bold] {settings.workspace}",
            f"[bold]Parallelism:[/bold] {settings.max_parallel_tasks}"
            f"    [bold]Max cycles:[/bold] {cycles}",
        ]
    )
    console.print(Panel(body, title="Furrow", border_style="green"))


def _print_summary(result: RunResult) -> None:
    """Print the terminal state and a summary table for a finished run.

    Args:
        result: The value returned by ``Orchestrator.run``.
    """
    if result.last_error:
        console.print(f"[red]{result.last_error}[/red]")
    if result.stopped_reason:
        console.print(f"[bold]{result.stopped_reason}[/bold]")
    table = Table(title="Run summary", title_justify="left", show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Cycles", str(result.cycles))
    table.add_row("Tasks completed", str(result.tasks_completed))
    table.add_row("Tasks failed", str(result.tasks_failed))
    if result.cycles_failed:
        table.add_row("Cycles failed", str(result.cycles_failed))
    console.print(table)
    if result.completed:
        console.print("[bold green]Goal complete.[/bold green]")
        raise SystemExit(EXIT_OK)
    console.print("[bold red]Run ended without completing the goal.[/bold red]")
    raise SystemExit(EXIT_FAILED)


@click.group()
@click.version_option(version=__version__, prog_name="furrow")
def main() -> None:
    """Autonomous coding agent that loops until your goal is met."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--provider",
    type=_PROVIDER_CHOICES,
    default=None,
    help="LLM provider to use; overrides the configured default.",
)
@click.option(
    "--model",
    default=None,
    help="Model to use for every role, overriding the per-role model settings.",
)
@click.option(
    "--workspace",
    default=None,
    help="Directory to work in; defaults to the configured workspace.",
)
@click.option(
    "--max-parallel",
    type=int,
    default=None,
    help="Maximum tasks executed in parallel; must be at least 1.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Maximum plan/execute cycles; 0 means unlimited.",
)
@click.option(
    "--test-command",
    default=None,
    help="Command used to run tests; auto-detected when unset.",
)
@click.option("--quiet", is_flag=True, default=False, help="Suppress panels and the spinner.")
@click.option(
    "--log-level",
    type=_LOG_LEVEL_CHOICES,
    default=None,
    help="Logging verbosity; defaults to the configured FURROW_LOG_LEVEL.",
)
def start(
    goal: str | None,
    provider: str | None,
    model: str | None,
    workspace: str | None,
    max_parallel: int | None,
    max_cycles: int | None,
    test_command: str | None,
    quiet: bool,
    log_level: str | None,
) -> None:
    """Run the agent loop until the goal is met, the budget runs out, or Ctrl-C.

    Exits 0 when the run completes, 1 when it finishes without completing the
    goal, and 130 when interrupted.
    """
    resolved_goal = goal or click.prompt("Enter your goal for Furrow")
    _validate_options(max_parallel, max_cycles, workspace)
    # get_settings(), never the module-level `settings` alias: it resolves the
    # same cached singleton but cannot be shadowed by import order elsewhere.
    settings = get_settings()
    if log_level is not None:
        settings.log_level = log_level
    # Applied here, not at import time: importing this module must not configure
    # logging for a host process, and the level must be live before any work runs.
    configure_logging(settings.log_level)
    effective_provider, effective_model = _configure(
        settings, provider, model, workspace, max_parallel, max_cycles, test_command
    )
    _render_banner(resolved_goal, effective_provider, effective_model, settings, quiet)

    from furrow.core.orchestrator import Orchestrator

    status = _NullStatus() if quiet else Status("Furrow is working...", console=console)
    on_event = None if quiet else _make_reporter()
    orchestrator = Orchestrator(goal=resolved_goal, settings=settings, on_event=on_event)
    try:
        with status:
            result = asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        raise SystemExit(EXIT_INTERRUPTED) from None
    except Exception as exc:
        console.print(f"[bold red]run failed:[/bold red] {type(exc).__name__}: {exc}")
        raise SystemExit(EXIT_FAILED) from None
    _print_summary(result)


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind; anything but loopback exposes an unauthenticated agent.",
)
@click.option("--port", default=8000, show_default=True, type=int, help="TCP port to listen on.")
@click.option(
    "--reload", "reload_enabled", is_flag=True, default=False, help="Reload on code changes."
)
def web(host: str, port: int, reload_enabled: bool) -> None:
    """Serve the browser UI that streams a run over a WebSocket."""
    # Deferred on purpose: furrow.web.server imports fastapi and uvicorn, which
    # a terminal-only session must never be made to pay for.
    from furrow.web.server import run

    run(host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    sys.exit(main())
