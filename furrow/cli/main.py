"""Command line entry point for the Furrow autonomous coding agent.

Wires configuration from env vars and a ``.env`` file (via ``reload_settings``),
applies CLI overrides on top, and drives either ``furrow start`` (run the agent
loop) or ``furrow web`` (serve the web UI).

Configuration precedence, highest to lowest:
    CLI flag  >  ``FURROW_*`` env var  >  ``.env`` file  >  built-in default.
"""
from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, Settings, reload_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, LLMConfigError, LLMError

__all__ = ["main"]

console = Console()

#: Cycle cap the CLI applies so the loop cannot spend forever against a paid API
#: when neither a CLI flag nor an env var sets one.
_DEFAULT_MAX_CYCLES = 10

#: Env vars that satisfy each provider's API key, surfaced in actionable errors.
_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    Provider.ANTHROPIC.value: ("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    Provider.OPENAI.value: ("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
}

#: Hosts that bind the loop interface only and are therefore safe by default.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _resolve_settings(
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
) -> Settings:
    """Build settings honoring config precedence: CLI flag > env > .env > default.

    Overrides are applied to a fresh ``reload_settings()`` result (never to the
    stale module-level singleton) via ``model_copy`` so the resulting object is
    handed straight to the :class:`~furrow.llm.LLMClient` the orchestrator uses.
    """
    settings = reload_settings()
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
        updates["worker_model"] = model
    if planner_model is not None:
        updates["planner_model"] = planner_model
    if worker_model is not None:
        updates["worker_model"] = worker_model
    if tester_model is not None:
        updates["tester_model"] = tester_model
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    elif settings.max_cycles <= 0:
        updates["max_cycles"] = _DEFAULT_MAX_CYCLES
    if max_parallel is not None:
        updates["max_parallel_tasks"] = max_parallel
    if workspace is not None:
        updates["workspace"] = Path(workspace)
    if updates:
        return settings.model_copy(update=updates)
    return settings


def _report_config_error(exc: LLMConfigError, settings: Settings) -> None:
    """Print an :class:`LLMConfigError` with the env var the user should set."""
    console.print(f"[red]Configuration error:[/red] {exc}")
    provider_name = settings.provider.value
    env_vars = _PROVIDER_ENV_VARS.get(provider_name)
    if env_vars:
        console.print(
            f"For provider '{provider_name}', set the environment variable "
            f"{', '.join(env_vars)}."
        )
    else:
        console.print(
            f"[yellow]Provider '{provider_name}' is not yet implemented, "
            "so no API key can be configured for it.[/yellow]"
        )


@click.group()
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Show full tracebacks for unexpected errors (expected LLM errors "
    "are always reported cleanly).",
)
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """Furrow: an autonomous coding agent with an infinite parallel dev loop.

    Configuration precedence (highest to lowest):
      CLI flag  >  FURROW_* env var  >  .env file  >  built-in default.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    "model",
    default=None,
    help="Override the coding model. Sets the general `model` and the "
    "`worker_model` (the model that writes code); `planner_model` and "
    "`tester_model` are left unchanged. Use --worker-model, --planner-model, "
    "or --tester-model to override those roles individually; they take "
    "precedence over --model.",
)
@click.option(
    "--planner-model",
    "planner_model",
    default=None,
    help="Override the planner (planning) model only.",
)
@click.option(
    "--worker-model",
    "worker_model",
    default=None,
    help="Override the worker (code-writing) model only.",
)
@click.option(
    "--tester-model",
    "tester_model",
    default=None,
    help="Override the tester model only.",
)
@click.option(
    "--max-cycles",
    "max_cycles",
    type=int,
    default=None,
    help="Cap plan/execute/test cycles to bound paid-API spend (default 10; "
    "0 = unlimited).",
)
@click.option(
    "--max-parallel",
    "max_parallel",
    type=int,
    default=None,
    help="Max concurrent worker tasks per cycle (default 5 or "
    "FURROW_MAX_PARALLEL_TASKS).",
)
@click.option(
    "--workspace",
    "workspace",
    type=click.Path(),
    default=None,
    help="Root directory the agent may read/write (default: current directory; "
    "FURROW_WORKSPACE).",
)
@click.pass_context
def start(
    ctx: click.Context,
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
) -> None:
    """Run the Furrow agent loop on GOAL."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    goal_text: str = goal if goal is not None else ""

    settings = _resolve_settings(
        model=model,
        planner_model=planner_model,
        worker_model=worker_model,
        tester_model=tester_model,
        max_cycles=max_cycles,
        max_parallel=max_parallel,
        workspace=workspace,
    )
    debug = bool(ctx.obj.get("debug")) if ctx.obj else False

    # Build the client with the resolved settings so every agent (planner,
    # worker, tester) reads the CLI/env overrides from its single source of truth.
    client = LLMClient(settings=settings)
    try:
        asyncio.run(Orchestrator(goal=goal_text, client=client, console=console).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        return
    except (LLMError, ValueError) as exc:
        if isinstance(exc, LLMConfigError):
            _report_config_error(exc, settings)
        else:
            console.print(f"[red]Error:[/red] {exc}")
        if debug:
            traceback.print_exc()
        raise SystemExit(1) from None
    except Exception:  # noqa: BLE001 - never leak a raw traceback unless asked
        console.print(
            "[red]Furrow failed unexpectedly. Re-run with --debug for the "
            "full traceback.[/red]"
        )
        if debug:
            traceback.print_exc()
        raise SystemExit(1) from None


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind address for the web UI (defaults to loopback 127.0.0.1).",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="Port for the web UI.",
)
@click.pass_context
def web(ctx: click.Context, host: str, port: int) -> None:
    """Serve the Furrow web UI."""
    # This endpoint accepts arbitrary goals and spends the user's API key, so
    # binding non-loopback interfaces by default would be a real hazard.
    if host not in _LOOPBACK_HOSTS:
        console.print(
            "[bold yellow]WARNING:[/bold yellow] serving on a non-loopback host "
            f"({host}); this exposes the agent and your LLM API key to the network."
        )
    from furrow.web.server import run as _run_web_server

    _run_web_server(host=host, port=port)


if __name__ == "__main__":
    main()
