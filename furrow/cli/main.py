"""Furrow's command line: ``furrow start`` runs the loop, ``furrow web`` the UI.

Two rules shape this module:

* Importing it does nothing. No event loop, no server, no global state is created
  at import time, so ``--version`` and ``--help`` stay instant and side-effect free.
* Every invocation builds its own :class:`~furrow.config.Settings`. ``FURROW_*``
  env vars and ``.env`` still apply (pydantic-settings reads them), explicit flags
  win on top, and the module-level ``furrow.config.settings`` singleton is never
  mutated -- so a run can never leak configuration into the next one.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()

PACKAGE_NAME = "furrow"
# Mirrors ``[project] version`` in pyproject.toml and is only used when the
# distribution metadata is missing (a source checkout that was never installed).
FALLBACK_VERSION = "0.1.0"

# Same defaults as ``furrow.web.server.run``; a UI that binds 0.0.0.0 by default
# publishes an unauthenticated agent to the whole network.
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000

PROVIDERS = [provider.value for provider in Provider]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# provider -> (Settings field with the key, fallback env vars, most specific first)
CREDENTIALS: dict[Provider, tuple[str, tuple[str, ...]]] = {
    Provider.ANTHROPIC: ("anthropic_api_key", ("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")),
    Provider.OPENAI: ("openai_api_key", ("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY")),
    Provider.OLLAMA: ("", ()),  # a local server needs no credentials
}


def get_version() -> str:
    """Resolve the version lazily: ``furrow`` is often run straight from a checkout."""
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return FALLBACK_VERSION
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else FALLBACK_VERSION


def build_settings(**overrides: Any) -> Settings:
    """Build the settings for a single invocation.

    ``FURROW_*`` env vars and ``.env`` still apply; ``overrides`` (the flags the
    user actually passed) take precedence. ``None`` means "not given" and is
    dropped, so an unset flag falls through to env/``.env`` instead of erasing it.
    """
    return Settings(**{name: value for name, value in overrides.items() if value is not None})


def _config_error(exc: ValidationError) -> click.ClickException:
    """Turn a pydantic validation failure into one short, actionable line."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
        for error in exc.errors()[:3]
    )
    return click.ClickException(f"invalid configuration: {problems}")


def _require_credentials(settings: Settings) -> None:
    """Fail fast (before any request) when the provider has no usable key."""
    field, env_names = CREDENTIALS[settings.provider]
    if not env_names:
        return  # a local provider needs no credentials
    # The field covers FURROW_* and .env; os.getenv covers the plain vendor vars.
    if (field and getattr(settings, field, None)) or any(os.getenv(name) for name in env_names):
        return
    raise click.ClickException(
        f"no API key found for provider '{settings.provider.value}'. "
        f"Set {' or '.join(env_names)} in your environment or in a .env file, "
        f"or run with --provider {Provider.OLLAMA.value} to use a local model."
    )


def _show_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """``--version`` callback: print and exit, resolving the version only when asked."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"{PACKAGE_NAME} {get_version()}")
    ctx.exit()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_show_version,
    help="Show the furrow version and exit.",
)
def main() -> None:
    """Furrow: an autonomous coding agent that keeps working until the goal is met."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--provider",
    type=click.Choice(PROVIDERS, case_sensitive=False),
    default=None,
    metavar="PROVIDER",
    help=f"LLM provider: {', '.join(PROVIDERS)}. "
    "Default: $FURROW_PROVIDER, else anthropic.",
)
@click.option(
    "--model",
    default=None,
    metavar="NAME",
    help="Model for every agent that has no role-specific override. "
    "Default: $FURROW_MODEL, else the Settings default.",
)
@click.option(
    "--planner-model",
    default=None,
    metavar="NAME",
    help="Model that writes the task plans. Default: $FURROW_PLANNER_MODEL, else claude-3-5-haiku.",
)
@click.option(
    "--worker-model",
    default=None,
    metavar="NAME",
    help="Model that implements the tasks. Default: $FURROW_WORKER_MODEL, else claude-3-5-sonnet.",
)
@click.option(
    "--tester-model",
    default=None,
    metavar="NAME",
    help="Model that judges the test output. Default: $FURROW_TESTER_MODEL, else claude-3-5-sonnet.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    metavar="N",
    help="Stop after N plan/execute/test cycles. Default: $FURROW_MAX_CYCLES, else 0 "
    "(run until the goal is met or a guard trips).",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=None,
    metavar="N",
    help="How many tasks of a cycle may run at once. Default: $FURROW_MAX_PARALLEL_TASKS, "
    "else 5 (0 means no limit).",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="DIR",
    help="Repository the agents read and write. Default: $FURROW_WORKSPACE, else the "
    "current directory.",
)
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default=None,
    metavar="LEVEL",
    help=f"Structlog level: {', '.join(LOG_LEVELS)}. Default: $FURROW_LOG_LEVEL, else INFO.",
)
def start(
    goal: str | None,
    provider: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    workspace: Path | None,
    log_level: str | None,
) -> None:
    """Run the development loop until the goal is met (or you press Ctrl-C).

    GOAL is the objective; when it is omitted you are prompted for one. Every
    other option only overrides the matching setting for this run.
    """
    goal = (goal or "").strip() or click.prompt("Enter your goal for Furrow")
    for hint, value in (("--max-cycles", max_cycles), ("--max-parallel-tasks", max_parallel_tasks)):
        if value is not None and value < 0:
            raise click.BadParameter("must be 0 or greater (0 means unlimited)", param_hint=hint)

    try:
        settings = build_settings(
            provider=Provider(provider.lower()) if provider else None,
            model=model,
            planner_model=planner_model,
            worker_model=worker_model,
            tester_model=tester_model,
            max_cycles=max_cycles,
            max_parallel_tasks=max_parallel_tasks,
            workspace=workspace,
            log_level=log_level.upper() if log_level else None,
        )
    except ValidationError as exc:  # e.g. FURROW_PROVIDER=bedrock in the environment
        raise _config_error(exc) from exc

    _require_credentials(settings)
    try:
        # The client carries the per-run models/keys (agents read them off it), the
        # orchestrator settings carry the loop's budget, workspace and log level.
        asyncio.run(
            Orchestrator(goal=goal, settings=settings, client=LLMClient(settings=settings)).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option(
    "--host",
    default=WEB_HOST,
    show_default=True,
    metavar="HOST",
    help="Interface to bind. Loopback only by default; use 0.0.0.0 to expose the "
    "unauthenticated server to your network.",
)
@click.option(
    "--port",
    type=int,
    default=WEB_PORT,
    show_default=True,
    metavar="PORT",
    help="TCP port to bind.",
)
def web(host: str, port: int) -> None:
    """Serve the web UI and its WebSocket API."""
    from furrow.web.server import run

    try:
        run(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow web server stopped by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()