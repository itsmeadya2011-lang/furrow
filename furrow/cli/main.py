"""Command line interface for furrow.

``furrow start`` runs the agent loop against a goal, ``furrow web`` serves the
browser monitoring UI, and the group itself only dispatches.

Exit codes: ``0`` success, ``1`` an unexpected runtime error (reported as a
single ``Error: ...`` line instead of a traceback), ``2`` a configuration problem
such as a missing API key, and ``130`` the conventional "terminated by SIGINT".
"""

from __future__ import annotations

import asyncio
import os
import sys
from importlib.metadata import PackageNotFoundError, version

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()

# Environment variable names for the pre-flight API key check, keyed by provider.
# Ollama is intentionally absent: it needs no key.
_API_KEY_ENV: dict[Provider, tuple[str, str]] = {
    Provider.ANTHROPIC: ("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    Provider.OPENAI: ("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
}

EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130


def _package_version() -> str:
    """Installed distribution version, or a marker when running from a checkout."""
    try:
        return version("furrow")
    except PackageNotFoundError:
        return "unknown (not installed as a distribution)"


def _apply_model_overrides(
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    """Push the ``--*-model`` flags into the global settings singleton.

    ``--model`` must overwrite every per-agent field as well as the generic
    fallback: the agents read ``planner_model``/``worker_model``/``tester_model``
    (``BaseAgent._model``) and always pass that value down to the client, so
    setting only ``settings.model`` would be a silent no-op.  The per-agent flags
    are applied afterwards, so they win over ``--model``.
    """
    if model:
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model


def _missing_api_key_message() -> str | None:
    """Return an actionable message when the configured provider has no API key."""
    provider = settings.provider
    names = _API_KEY_ENV.get(provider)
    if names is None:
        return None
    configured = (
        settings.anthropic_api_key
        if provider == Provider.ANTHROPIC
        else settings.openai_api_key
    )
    if configured or any(os.getenv(name) for name in names):
        return None
    prefixed, plain = names
    return (
        f"no API key configured for provider '{provider.value}'. Set "
        f"{prefixed} (the Furrow setting) or {plain} (the environment "
        f"variable), then try again."
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=_package_version(), prog_name="furrow")
def main() -> None:
    """Furrow: an autonomous coding agent that loops until the goal is met.

    Run `furrow start "add a REST API"` to work in the terminal, or `furrow web`
    to watch the same loop from a browser.
    """


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    default=None,
    help="Model for every agent (planner, worker and tester); overrides the configured "
    "model, planner_model, worker_model and tester_model settings.",
)
@click.option("--planner-model", default=None, help="Model for the planner agent only.")
@click.option("--worker-model", default=None, help="Model for worker agents only.")
@click.option("--tester-model", default=None, help="Model for the tester agent only.")
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    """Run the agent loop against GOAL, prompting for one when it is omitted."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    _apply_model_overrides(model, planner_model, worker_model, tester_model)
    problem = _missing_api_key_message()
    if problem is not None:
        console.print(f"[red]Error:[/red] {problem}")
        sys.exit(EXIT_CONFIG)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(EXIT_ERROR)


@main.command()
@click.option("--host", default=None, help="Interface to bind (default: the host setting).")
@click.option(
    "--port",
    default=None,
    type=int,
    help="Port to listen on (default: the port setting).",
)
def web(host: str | None, port: int | None) -> None:
    """Serve the browser monitoring UI."""
    from furrow.web.server import run

    try:
        run(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow web server stopped by user.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()