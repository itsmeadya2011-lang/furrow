"""Command-line entry point for the Furrow autonomous coding agent."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.config import (
    Provider,
    RunResult,
    Settings,
    TaskStatus,
)
from furrow.config import (
    settings as global_settings,
)
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()

#: Human-readable labels for each ``RunResult.stopped_reason`` value.
_HALT_LABELS: dict[str, str] = {
    "goal_complete": "Goal Complete",
    "max_cycles": "Max Cycles Reached",
    "stopped": "Stopped by Request",
    "error": "Error",
}


def _provider_names() -> list[str]:
    """Return the lowercase value of every ``Provider`` member."""
    return [member.value for member in Provider]


def _build_settings(
    *,
    provider: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    workspace: str | None,
) -> Settings:
    """Construct a fresh ``Settings`` with CLI overrides applied.

    A brand-new ``Settings`` is returned rather than mutating the
    module-level ``settings`` singleton, so model/provider overrides never
    leak into other components that hold a reference to the global.  Each
    override is only applied when the corresponding flag was supplied,
    allowing ``FURROW_*`` env vars and ``.env`` to fill the gaps.
    """
    overrides: dict[str, Any] = {}
    if provider is not None:
        overrides["provider"] = Provider(provider)
    if model is not None:
        overrides["model"] = model
    if planner_model is not None:
        overrides["planner_model"] = planner_model
    if worker_model is not None:
        overrides["worker_model"] = worker_model
    if tester_model is not None:
        overrides["tester_model"] = tester_model
    if workspace is not None:
        overrides["workspace"] = Path(workspace)
    return Settings(**overrides)


def _excerpt(text: str, limit: int = 160) -> str:
    """Collapse *text* to a single line of at most *limit* characters."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: max(1, limit - 3)] + "..."


def _print_config(
    config: Settings,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
) -> None:
    """Display the resolved run configuration (used by ``--dry-run``)."""
    effective_cycles = (
        max_cycles if max_cycles is not None else config.max_cycles
    )
    effective_parallel = (
        max_parallel_tasks
        if max_parallel_tasks is not None
        else config.max_parallel_tasks
    )

    table = Table(
        title="Resolved Configuration",
        show_header=True,
        header_style="cyan",
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("provider", str(config.provider))
    table.add_row("model", config.model)
    table.add_row("planner_model", config.planner_model)
    table.add_row("worker_model", config.worker_model)
    table.add_row("tester_model", config.tester_model)
    table.add_row("workspace", str(config.workspace))
    table.add_row("max_cycles", str(effective_cycles))
    table.add_row("max_parallel_tasks", str(effective_parallel))
    table.add_row("test_timeout", str(config.test_timeout))
    table.add_row("test_command", str(config.test_command))
    console.print(table)
    console.print("\n[dim]Dry run -- configuration resolved, no agents invoked.[/dim]")


def _print_summary(result: RunResult) -> None:
    """Render a final summary table from the ``RunResult`` of ``run()``."""
    completed = sum(
        1 for t in result.tasks if t.status is TaskStatus.COMPLETED
    )
    failed = sum(1 for t in result.tasks if t.status is TaskStatus.FAILED)
    skipped = sum(1 for t in result.tasks if t.status is TaskStatus.SKIPPED)
    total = len(result.tasks)

    label = _HALT_LABELS.get(result.stopped_reason, result.stopped_reason)
    if result.stopped_reason == "goal_complete":
        color = "green"
    elif result.stopped_reason == "error":
        color = "red"
    else:
        color = "yellow"

    console.print()
    console.print(
        Panel.fit(
            (
                f"[bold {color}]{label}[/]\n"
                f"[dim]Goal:[/dim]    {result.goal}\n"
                f"[dim]Cycles:[/dim] {result.cycles}\n"
                f"[dim]Tasks:[/dim]   "
                f"{completed} completed, {failed} failed, "
                f"{skipped} skipped ({total} total)"
            ),
            title="Furrow Summary",
            border_style=color,
        )
    )

    if result.test_results:
        last = result.test_results[-1]
        badge = "[green]PASSED[/]" if last.passed else "[red]FAILED[/]"
        console.print(
            f"\n[bold]Last test result:[/bold] {badge} "
            f"-- {_excerpt(last.summary, 160)}"
        )
        for fail in last.failures[:5]:
            console.print(f"  [red]*[/red] {_excerpt(fail, 120)}")
        if len(last.failures) > 5:
            console.print(
                f"  [dim]... and {len(last.failures) - 5} more[/dim]"
            )

    if result.reports:
        console.print()
        table = Table(
            title="Cycle Reports",
            header_style="cyan",
        )
        table.add_column("Cycle", style="dim")
        table.add_column("Planned", justify="right")
        table.add_column("Completed", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Tests")
        table.add_column("Summary")
        for report in result.reports:
            if report.tests_passed is True:
                tests = "[green]yes[/]"
            elif report.tests_passed is False:
                tests = "[red]no[/]"
            else:
                tests = "[dim]--[/]"
            table.add_row(
                str(report.cycle),
                str(report.planned),
                str(report.completed),
                str(report.failed),
                tests,
                _excerpt(report.summary, 80),
            )
        console.print(table)


@click.group()
def main() -> None:
    """Furrow -- an autonomous coding agent."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    default=None,
    help="Override the default LLM model.",
)
@click.option(
    "--provider",
    type=click.Choice(_provider_names(), case_sensitive=False),
    default=None,
    help="LLM provider to use (overrides FURROW_PROVIDER).",
)
@click.option(
    "--max-cycles",
    "max_cycles",
    type=int,
    default=None,
    help="Maximum plan/execute/test cycles; 0 = unlimited.",
)
@click.option(
    "--max-parallel-tasks",
    "max_parallel_tasks",
    type=int,
    default=None,
    help="Maximum number of tasks to run in parallel.",
)
@click.option(
    "--workspace",
    type=click.Path(),
    default=None,
    help="Working directory for the run.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Resolve configuration and exit without running.",
)
@click.option(
    "--planner-model",
    default=None,
    help="Override the planner LLM model.",
)
@click.option(
    "--worker-model",
    default=None,
    help="Override the worker LLM model.",
)
@click.option(
    "--tester-model",
    default=None,
    help="Override the tester LLM model.",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    workspace: str | None,
    dry_run: bool,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    """Start a Furrow run for GOAL (prompts interactively if omitted)."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Build a local Settings so model/provider overrides never touch the
    # shared global singleton.  LLMClient and the agents read model/provider
    # from client.settings, so this keeps the global ``settings`` clean.
    config = _build_settings(
        provider=provider,
        model=model,
        planner_model=planner_model,
        worker_model=worker_model,
        tester_model=tester_model,
        workspace=workspace,
    )

    if dry_run:
        _print_config(config, max_cycles, max_parallel_tasks)
        return

    # The Orchestrator reads ``max_parallel_tasks`` from the module-level
    # ``settings`` singleton (see furrow.core.orchestrator), not from the
    # LLMClient.  We set it there deliberately because there is no constructor
    # parameter for it; all other overrides are passed via *config* above.
    if max_parallel_tasks is not None:
        global_settings.max_parallel_tasks = max_parallel_tasks

    client = LLMClient(settings=config)

    try:
        result = asyncio.run(
            Orchestrator(
                goal=goal,
                client=client,
                max_cycles=max_cycles,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)

    _print_summary(result)


@main.command()
@click.option(
    "--host",
    default=global_settings.web_host,
    show_default=True,
    help="Host to bind the web server to.",
)
@click.option(
    "--port",
    default=global_settings.web_port,
    show_default=True,
    type=int,
    help="Port to bind the web server to.",
)
def web(host: str, port: int) -> None:
    """Start the Furrow web UI server."""
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
