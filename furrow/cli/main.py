"""Furrow command line interface."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()

F = TypeVar("F", bound=Callable[..., Any])

PROVIDERS = ("anthropic", "openai", "ollama")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# Environment variables consulted for each provider, in priority order.
# The FURROW_ prefixed names are what pydantic-settings reads (env_prefix="FURROW_"),
# the bare names are the conventional fallbacks honoured by furrow.llm.LLMClient.
CREDENTIAL_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
}

# Roles that can be pointed at a dedicated model via a ``start`` flag.
MODEL_ROLES = ("planner_model", "worker_model", "tester_model")


def _resolve_version() -> str:
    """Return the installed version, falling back without ever raising."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("furrow")
        except PackageNotFoundError:
            pass
    except ImportError:  # pragma: no cover - Python < 3.8 only
        pass
    import furrow

    return str(getattr(furrow, "__version__", "0.0.0+unknown"))


def _provider_name(value: object) -> str:
    """Normalise a provider value (plain string or enum) to its string form."""
    return str(getattr(value, "value", value))


def _provider_default() -> str:
    name = _provider_name(settings.provider)
    return name if name in PROVIDERS else PROVIDERS[0]


def _log_level_default() -> str:
    level = str(settings.log_level or LOG_LEVELS[1]).upper()
    return level if level in LOG_LEVELS else LOG_LEVELS[1]


def _credential() -> str | None:
    """Return the configured credential for the active provider, else None."""
    name = _provider_name(settings.provider)
    if name == "anthropic":
        configured = settings.anthropic_api_key
    elif name == "openai":
        configured = settings.openai_api_key
    else:
        # ollama runs locally: the base url is all that is needed.
        return settings.ollama_base_url
    if configured:
        return configured
    # Settings resolves .env/FURROW_* at import time; re-check the environment so a
    # key exported after import (or a changed env_prefix) is still honoured.
    for env_name in CREDENTIAL_ENV.get(name, ()):
        value = os.getenv(env_name)
        if value:
            return value
    return None


def _require_credential() -> None:
    """Fail fast with an actionable message when the provider cannot be used.

    Called from the subcommands rather than the group callback, so ``--help`` and
    ``--version`` keep working on a machine without credentials.
    """
    name = _provider_name(settings.provider)
    if name not in CREDENTIAL_ENV or _credential():
        return
    env_names = " or ".join(CREDENTIAL_ENV[name])
    raise click.ClickException(
        f"no API key configured for provider {name!r}. "
        f"Set {env_names} (environment or .env file), "
        "or pick another provider with --provider."
    )


def _configure_logging(level: str) -> None:
    settings.log_level = level
    logging.getLogger("furrow").setLevel(getattr(logging, level.upper(), logging.INFO))


def _global_options(*, show_defaults: bool) -> Callable[[F], F]:
    """Build the option decorators shared by the group and its subcommands.

    They are attached to the group (where the configured defaults are shown) and to
    every subcommand, so ``furrow -w DIR start GOAL`` and ``furrow start -w DIR GOAL``
    both work. A value given to the subcommand wins over the group value.
    """

    def decorate(fn: F) -> F:
        options = [
            click.option(
                "--workspace",
                "-w",
                "workspace",
                type=click.Path(file_okay=False, path_type=Path),
                default=settings.workspace if show_defaults else None,
                show_default=show_defaults,
                metavar="PATH",
                help="Directory the agents work in.",
            ),
            click.option(
                "--max-parallel-tasks",
                type=click.IntRange(1, 32),
                default=settings.max_parallel_tasks if show_defaults else None,
                show_default=show_defaults,
                help="Maximum tasks executed in parallel per cycle (1-32).",
            ),
            click.option(
                "--max-cycles",
                type=click.IntRange(0),
                default=settings.max_cycles if show_defaults else None,
                show_default=show_defaults,
                help="Stop after this many plan/execute/test cycles. 0 means unlimited.",
            ),
            click.option(
                "--provider",
                type=click.Choice(PROVIDERS),
                default=_provider_default() if show_defaults else None,
                show_default=show_defaults,
                help="LLM provider to use.",
            ),
            click.option(
                "--log-level",
                type=click.Choice(LOG_LEVELS),
                default=_log_level_default() if show_defaults else None,
                show_default=show_defaults,
                help="Logging verbosity for the furrow loggers.",
            ),
        ]
        for option in reversed(options):
            fn = option(fn)
        return fn

    return decorate


def _apply_globals(
    workspace: Path | None,
    max_parallel_tasks: int | None,
    max_cycles: int | None,
    provider: str | None,
    log_level: str | None,
) -> None:
    """Write explicitly provided global options into the shared settings object."""
    if provider is not None:
        settings.provider = provider
    if workspace is not None:
        settings.workspace = workspace
    if max_parallel_tasks is not None:
        settings.max_parallel_tasks = max_parallel_tasks
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if log_level is not None:
        _configure_logging(log_level)


def _apply_model_overrides(
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    """Apply ``--model`` plus per-role overrides to the shared settings object.

    Precedence: an explicit ``--planner-model``/``--worker-model``/``--tester-model``
    always wins; otherwise ``--model`` sets the default model *and* every role, because
    the agents read ``planner_model``/``worker_model``/``tester_model`` rather than
    ``model``.
    """
    if model:
        settings.model = model
    explicit = {
        "planner_model": planner_model,
        "worker_model": worker_model,
        "tester_model": tester_model,
    }
    for role in MODEL_ROLES:
        if explicit[role]:
            setattr(settings, role, explicit[role])
        elif model:
            setattr(settings, role, model)


def _debug_enabled() -> bool:
    return os.getenv("FURROW_TRACEBACK", "").strip().lower() not in ("", "0", "false", "no")


def _fail(exc: Exception) -> NoReturn:
    """Report an expected failure without a raw traceback and exit non-zero."""
    if _debug_enabled():
        raise exc
    console.print(f"[bold red]Error:[/bold red] {type(exc).__name__}: {exc}")
    console.print("[dim]Set FURROW_TRACEBACK=1 to see the full traceback.[/dim]")
    sys.exit(1)


def _accepts_kwarg(func: object, name: str) -> bool:
    """Return True when ``func`` can be called with the keyword argument ``name``."""
    try:
        signature = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(name)
    return parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@_global_options(show_defaults=True)
@click.version_option(version=_resolve_version(), prog_name="furrow")
@click.pass_context
def main(
    ctx: click.Context,
    workspace: Path | None,
    max_parallel_tasks: int | None,
    max_cycles: int | None,
    provider: str | None,
    log_level: str | None,
) -> None:
    """Furrow - autonomous coding agent with an infinite parallel development loop."""
    _apply_globals(workspace, max_parallel_tasks, max_cycles, provider, log_level)
    ctx.obj = settings


@main.command()
@click.argument("goal", required=False)
@_global_options(show_defaults=False)
@click.option(
    "--model",
    default=None,
    metavar="NAME",
    help="Model for every role: sets the default model and, unless a per-role flag is "
    "given, the planner/worker/tester too. An explicit --planner-model, --worker-model "
    "or --tester-model always wins over --model.",
)
@click.option(
    "--planner-model",
    default=None,
    metavar="NAME",
    help="Model for the planner agent only (overrides --model).",
)
@click.option(
    "--worker-model",
    default=None,
    metavar="NAME",
    help="Model for the worker agent only (overrides --model).",
)
@click.option(
    "--tester-model",
    default=None,
    metavar="NAME",
    help="Model for the tester agent only (overrides --model).",
)
def start(
    goal: str | None,
    workspace: Path | None,
    max_parallel_tasks: int | None,
    max_cycles: int | None,
    provider: str | None,
    log_level: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    """Plan, execute and test until GOAL is met.

    \b
    Examples:
        furrow start "Add JWT auth to the API"
        furrow --provider openai --max-parallel-tasks 8 start "Refactor the parser"
        furrow start --model gpt-4o --tester-model gpt-4o-mini "Fix flaky tests"
    """
    _apply_globals(workspace, max_parallel_tasks, max_cycles, provider, log_level)
    if not goal:
        goal = click.prompt("Enter your goal for Furrow", type=str)
    _require_credential()
    _apply_model_overrides(model, planner_model, worker_model, tester_model)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except (click.ClickException, click.Abort, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, never a traceback
        _fail(exc)


@main.command()
@_global_options(show_defaults=False)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind. Use 0.0.0.0 to accept connections from other hosts.",
)
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=8000,
    show_default=True,
    help="TCP port to listen on.",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Reload on code changes when the server factory supports it.",
)
def web(
    host: str,
    port: int,
    reload: bool,
    workspace: Path | None,
    max_parallel_tasks: int | None,
    max_cycles: int | None,
    provider: str | None,
    log_level: str | None,
) -> None:
    """Serve the web UI and websocket API."""
    from furrow.web.server import run

    _apply_globals(workspace, max_parallel_tasks, max_cycles, provider, log_level)
    _require_credential()
    kwargs: dict[str, Any] = {"host": host, "port": port}
    if reload:
        if _accepts_kwarg(run, "reload"):
            kwargs["reload"] = True
        else:
            console.print(
                "[yellow]--reload is not supported by furrow.web.server.run(); ignoring.[/yellow]"
            )
    console.print(f"[dim]Starting Furrow web UI on http://{host}:{port}[/dim]")
    try:
        run(**kwargs)
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Web server stopped by user.[/yellow]")
        sys.exit(0)
    except (click.ClickException, click.Abort, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, never a traceback
        _fail(exc)


if __name__ == "__main__":
    main()
