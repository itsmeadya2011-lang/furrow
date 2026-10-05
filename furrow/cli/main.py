from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, Settings, get_settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()

PROVIDERS = [provider.value for provider in Provider]

# Providers that need a key: (Settings property holding it, environment variable).
API_KEYS: dict[Provider, tuple[str, str]] = {
    Provider.ANTHROPIC: ("anthropic_api_key_text", "ANTHROPIC_API_KEY"),
    Provider.OPENAI: ("openai_api_key_text", "OPENAI_API_KEY"),
}


def missing_api_key(provider: Provider, settings: Settings) -> str | None:
    """Return the environment variable ``provider`` needs but has no key for.

    Returns ``None`` when a key is available, or for providers (ollama) that
    need none at all. Checking up front turns a late ``ValueError`` from inside
    the LLM client into an actionable CLI error.
    """
    required = API_KEYS.get(provider)
    if required is None:
        return None
    attr, env_var = required
    if getattr(settings, attr, None) or os.environ.get(env_var):
        return None
    return env_var


def build_settings(
    model: str | None = None,
    provider: str | None = None,
    max_cycles: int | None = None,
    max_parallel: int | None = None,
    test_command: str | None = None,
) -> Settings:
    """Return a per-invocation copy of the settings with CLI overrides applied.

    Only the options the user actually passed are applied, and the process-wide
    ``Settings`` singleton is never mutated: clients built elsewhere (or already
    constructed) must keep seeing the settings they were created with.
    """
    update: dict[str, Any] = {}
    if model:
        update["model"] = model
    if provider:
        update["provider"] = Provider(provider)
    if max_cycles is not None:
        update["max_cycles"] = max_cycles
    if max_parallel is not None:
        update["max_parallel_tasks"] = max_parallel
    if test_command:
        update["test_command"] = test_command
    return get_settings().model_copy(update=update)


@click.group()
def main() -> None:
    """Furrow: an autonomous coding agent with a parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model.")
@click.option(
    "--provider",
    type=click.Choice(PROVIDERS),
    default=None,
    help="Override LLM provider.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after N cycles (default: unlimited).",
)
@click.option(
    "--max-parallel",
    type=int,
    default=None,
    help="Run at most N tasks concurrently.",
)
@click.option(
    "--test-command",
    default=None,
    help="Shell command used to run the test suite.",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    test_command: str | None,
) -> None:
    """Run Furrow against GOAL until the goal is reached."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    settings = build_settings(
        model=model,
        provider=provider,
        max_cycles=max_cycles,
        max_parallel=max_parallel,
        test_command=test_command,
    )

    if provider:
        env_var = missing_api_key(Provider(provider), settings)
        if env_var is not None:
            raise click.UsageError(
                f"--provider {provider} needs an API key. Set {env_var} in your "
                "environment (or configure the provider key in .env) and retry."
            )

    client = LLMClient(settings=settings)
    try:
        asyncio.run(Orchestrator(goal=goal, client=client).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address.")
@click.option("--port", type=int, default=8000, show_default=True, help="Bind port.")
def web(host: str, port: int) -> None:
    """Serve the browser UI."""
    from furrow.web.server import run

    console.print(f"[bold green]Furrow[/bold green] web UI: http://{host}:{port}")
    run(host=host, port=port)


if __name__ == "__main__":
    main()
