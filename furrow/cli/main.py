from __future__ import annotations

import asyncio
import sys
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()

PROVIDER_CHOICES = [provider.value for provider in Provider]

#: Agents name their role model explicitly when they call ``LLMClient.complete``
#: (``PlannerAgent`` passes ``planner_model``, ``WorkerAgent`` ``worker_model``,
#: ``TesterAgent`` ``tester_model``), so ``Settings.model`` is only ever the
#: fallback for a caller that omits the argument. A user-supplied model has
#: therefore to be expanded across every role, otherwise ``--model`` is accepted
#: and then silently ignored by all three agents.
#: Mirrored by ``furrow.web.server.settings_for_model`` - the two front ends each
#: build their own copy, so see the note there on why this is not one helper.
ROLE_MODEL_FIELDS = ("planner_model", "worker_model", "tester_model")


def build_run_settings(
    model: str | None = None,
    provider: str | None = None,
    max_cycles: int | None = None,
    max_parallel_tasks: int | None = None,
) -> Settings:
    """Return a per-invocation copy of the global settings.

    ``furrow.config.settings`` is a process-wide singleton, so writing to it
    would leak a ``--model`` override into every other consumer in the process.
    Only options the user actually typed are applied: an omitted option must
    leave the env-derived value alone, and ``0`` is a real value ("unlimited")
    rather than a stand-in for "not set".

    A supplied ``--model`` also replaces the three role models, so the flag
    reaches the agents instead of stopping at the unused ``model`` field. With no
    ``--model`` the role models are not in ``updates`` at all, which leaves the
    deliberate per-role defaults in :class:`~furrow.config.Settings` intact.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
        for field in ROLE_MODEL_FIELDS:
            updates[field] = model
    if provider is not None:
        updates["provider"] = Provider(provider)
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if max_parallel_tasks is not None:
        updates["max_parallel_tasks"] = max_parallel_tasks
    return settings.model_copy(update=updates)


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the LLM model for this run")
@click.option(
    "--provider",
    type=click.Choice(PROVIDER_CHOICES),
    default=None,
    help="Override the LLM provider for this run",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after this many cycles (0 means unlimited)",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=None,
    help="Cap on tasks executed concurrently within a cycle",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    run_settings = build_run_settings(
        model=model,
        provider=provider,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
    )
    try:
        asyncio.run(Orchestrator(goal=goal, settings=run_settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
