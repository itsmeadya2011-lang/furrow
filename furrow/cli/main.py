from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()

EXIT_ERROR = 1
EXIT_CONFIG = 2

# Env var holding the API key for each key-authenticated provider.
API_KEY_ENV_VARS: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
}


@click.group()
def main() -> None:
    pass


def _missing_api_key(run_settings: Settings) -> str | None:
    """Return the env var expected to hold the API key, or None when configured."""
    env_var = API_KEY_ENV_VARS.get(run_settings.provider)
    if env_var is None:  # e.g. ollama, which needs no key
        return None
    configured = {
        Provider.ANTHROPIC: run_settings.anthropic_api_key,
        Provider.OPENAI: run_settings.openai_api_key,
    }[run_settings.provider]
    if configured or os.getenv(env_var):
        return None
    return env_var


def _build_orchestrator(goal: str, client: LLMClient, run_settings: Settings) -> Orchestrator:
    """Create the orchestrator, passing only kwargs the installed class accepts.

    ``settings`` is a newer constructor parameter; inspecting the signature keeps
    the CLI working against older orchestrators.
    """
    kwargs: dict[str, Any] = {"goal": goal, "client": client}
    if "settings" in inspect.signature(Orchestrator).parameters:
        kwargs["settings"] = run_settings
    return Orchestrator(**kwargs)


def _exit_unless_succeeded(outcome: Any) -> None:
    """Exit non-zero when the orchestrator reports the goal was not completed.

    The loop deliberately survives per-task and planner failures and simply reports
    how it ended, so without this check a total failure still exited 0 and a caller
    (CI, the scheduled-task runner) could not tell success from death. An older
    orchestrator returns ``None``, which is treated as success, matching the
    signature guards above.
    """
    if getattr(outcome, "succeeded", True):
        return
    stop_reason = getattr(outcome, "stop_reason", "unknown reason")
    completed = getattr(outcome, "completed", 0)
    cycles = getattr(outcome, "cycles", 0)
    console.print(
        f"[red]Furrow did not complete the goal:[/red] {stop_reason} "
        f"({completed} task(s) completed over {cycles} cycle(s))"
    )
    sys.exit(EXIT_ERROR)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the LLM model.")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider], case_sensitive=False),
    default=None,
    help="Override the LLM provider.",
)
@click.option("--max-cycles", type=int, default=None, help="Max cycles; 0 means unlimited.")
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory the run operates in.",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    workspace: Path | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    goal = (goal or "").strip()
    if not goal:
        console.print("[red]Goal cannot be empty.[/red]")
        sys.exit(EXIT_CONFIG)

    if max_cycles is not None and max_cycles < 0:
        console.print("[red]--max-cycles must be 0 or greater (0 means unlimited).[/red]")
        sys.exit(EXIT_CONFIG)

    # `settings` is a module-level singleton read by every agent, so overriding
    # fields in place is how CLI options are applied. The client is then built
    # explicitly from it, so a --model override definitely reaches the LLM calls
    # instead of depending on the orchestrator to construct its own client.
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider.lower())
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace is not None:
        settings.workspace = workspace

    try:
        client = LLMClient(settings=settings)
        # Fail fast with an actionable message rather than a traceback from deep
        # inside the first LLM call. Skipped when the client is substituted (e.g.
        # in tests), since a replacement may not need credentials.
        if type(client) is LLMClient:
            missing = _missing_api_key(settings)
            if missing:
                console.print(f"[red]Missing API key: {missing} is not set.[/red]")
                console.print(
                    f"  Set {missing} in your environment (or add {missing}=... to a .env "
                    f"file), or pick a provider that needs no key with --provider."
                )
                sys.exit(EXIT_CONFIG)
        orchestrator = _build_orchestrator(goal, client, settings)
        outcome = asyncio.run(orchestrator.run())
        _exit_unless_succeeded(outcome)
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except ValueError as exc:
        # Configuration problems, e.g. LLMClient("ANTHROPIC_API_KEY is not set")
        # or an unparseable plan from the planner agent.
        console.print(f"[red]Configuration error:[/red] {exc}")
        if os.getenv("FURROW_DEBUG"):
            console.print_exception()
        sys.exit(EXIT_CONFIG)
    # Deliberate last-resort net: any failure from the orchestrator or an agent
    # must reach the user as one readable line, never as a raw traceback.
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Furrow failed:[/red] {exc}")
        console.print("  Set FURROW_DEBUG=1 to re-run with the full traceback.")
        if os.getenv("FURROW_DEBUG"):
            console.print_exception()
        sys.exit(EXIT_ERROR)


@main.command()
@click.option("--host", default=None, help="Bind address (default: $FURROW_HOST or 0.0.0.0).")
@click.option("--port", type=int, default=None, help="Bind port (default: $FURROW_PORT or 8000).")
def web(host: str | None, port: int | None) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
