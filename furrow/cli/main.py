"""Command line interface for Furrow.

Every command is importable without side effects: heavy or optional imports
(``furrow.web.server``, the agents, the LLM clients) happen inside the command
body, and the global settings singleton is never mutated -- CLI overrides are
folded into a per-invocation copy with ``Settings.model_copy``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import signal
import sys
from collections.abc import Iterator, Mapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from pathlib import Path
from types import FrameType
from typing import Any, Final

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.config import Plan, Provider, Settings
from furrow.config import settings as default_settings
from furrow.core import Event, EventType, Orchestrator

console = Console()

#: The run finished, or was stopped cleanly on request.
EXIT_OK: Final[int] = 0
#: The run ended without reaching the goal, or blew up.
EXIT_ERROR: Final[int] = 1
#: The run was aborted before the loop could finish.
EXIT_INTERRUPTED: Final[int] = 130

#: Used when the distribution metadata is missing (plain source checkout).
FALLBACK_VERSION: Final[str] = "0.1.0"

#: Mirrors of ``furrow.web.server`` defaults so that module stays lazily imported.
DEFAULT_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 8000

#: Task statuses as written by the orchestrator, used for counting and colouring.
COMPLETED: Final[str] = "completed"
FAILED: Final[str] = "failed"

#: Accepted ``--provider`` values, taken straight from the config enum.
PROVIDER_CHOICES: Final[tuple[str, ...]] = tuple(item.value for item in Provider)


# --------------------------------------------------------------------- settings


def _build_settings(
    model: str | None = None,
    provider: str | None = None,
    workspace: Path | str | None = None,
    max_cycles: int | None = None,
    max_parallel: int | None = None,
    test_command: str | None = None,
    test_timeout: float | None = None,
) -> Settings:
    """Return a copy of the global settings with the passed CLI overrides applied.

    Only the options the user actually supplied are copied over, so environment
    variables and ``.env`` values keep working for everything else. The global
    singleton is never mutated.
    """
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if provider:
        overrides["provider"] = Provider(provider)
    if workspace is not None:
        overrides["workspace"] = Path(workspace)
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel
    if test_command:
        overrides["test_command"] = test_command
    if test_timeout is not None:
        overrides["test_timeout"] = test_timeout
    return default_settings.model_copy(update=overrides)


def _settings_options(func: Any) -> Any:
    """Attach the shared settings overrides to a command."""
    options = [
        click.option("--model", default=None, help="LLM model to run the loop with."),
        click.option(
            "--provider",
            type=click.Choice(PROVIDER_CHOICES, case_sensitive=False),
            default=None,
            help="LLM provider.",
        ),
        click.option(
            "--workspace",
            type=click.Path(file_okay=False, path_type=Path),
            default=None,
            help="Workspace root the agents read and write.",
        ),
        click.option(
            "--cycles",
            "max_cycles",
            type=int,
            default=None,
            help="Cycle budget; 0 means unlimited.",
        ),
        click.option(
            "--max-parallel",
            "max_parallel",
            type=click.IntRange(min=1),
            default=None,
            help="Tasks executed at the same time.",
        ),
        click.option("--test-command", default=None, help="Command the tester agent runs."),
        click.option(
            "--test-timeout",
            type=click.FloatRange(min=0.0),
            default=None,
            help="Seconds allowed for a test run.",
        ),
    ]
    for decorator in reversed(options):
        func = decorator(func)
    return func


def _describe(settings: Settings) -> str:
    """Return a one line recap of the settings a run will use."""
    return (
        f"[dim]{settings.provider.value} · {settings.model} · "
        f"{settings.workspace} · max {settings.max_parallel_tasks} parallel"
        f"{' · unlimited cycles' if settings.max_cycles <= 0 else ''}[/dim]"
    )


def _resolve_goal(goal: str | None) -> str:
    """Return the goal argument, prompting for one when it is missing."""
    candidate = (goal or click.prompt("Enter your goal for Furrow")).strip()
    if not candidate:
        raise click.UsageError("A goal is required.")
    return candidate


# ------------------------------------------------------------------- rendering


def _as_event_type(value: object) -> EventType | None:
    """Coerce an event type that may have been handed over as a plain string."""
    try:
        return EventType(value)
    except ValueError:
        return None


def _oneline(text: object) -> str:
    """Flatten a message so one event never turns into several console lines."""
    return " ".join(str(text or "").split())


class EventRenderer:
    """Renders orchestrator events as compact, colourised progress lines.

    It is a plain callable, so it can be passed straight to
    ``Orchestrator(on_event=...)``. ``rich`` drops the colour automatically when
    stdout is not a terminal, so piped output stays readable.
    """

    def __init__(self, target: Console | None = None) -> None:
        self.console = target if target is not None else console
        self.errors = 0
        self.interrupted = False
        self.stopped = False

    def __call__(self, event: Event) -> None:
        """Print one orchestrator event."""
        kind = _as_event_type(event.type)
        data = event.data if isinstance(event.data, dict) else {}
        prefix = f"[dim]c{event.cycle}[/dim] " if event.cycle else ""
        message = _oneline(event.message)

        if kind is EventType.CYCLE_START:
            self.console.print(f"\n[bold cyan]── cycle {event.cycle} ──[/bold cyan]")
        elif kind is EventType.PLAN:
            planned = len(data.get("tasks") or [])
            self.console.print(f"{prefix}[cyan]plan[/cyan] [dim]{planned} task(s)[/dim] {message}")
        elif kind is EventType.TASK_START:
            self.console.print(f"{prefix}[yellow]▶ {self._task_label(data, message)}[/yellow]")
        elif kind is EventType.TASK_END:
            raw_task = data.get("task")
            task = raw_task if isinstance(raw_task, dict) else {}
            style = "green" if task.get("status") == COMPLETED else "red"
            mark = "✓" if style == "green" else "✗"
            self.console.print(
                f"{prefix}[{style}]{mark} {self._task_label(data, message)}[/{style}]"
            )
        elif kind is EventType.TEST_START:
            self.console.print(f"{prefix}[dim]tests running: {message}[/dim]")
        elif kind is EventType.TEST_END:
            style = "green" if data.get("passed") else "red"
            verdict = "passed" if data.get("passed") else "failed"
            self.console.print(f"{prefix}[{style}]tests {verdict}: {message}[/{style}]")
        elif kind is EventType.CYCLE_END:
            self.console.print(f"{prefix}[dim]cycle {event.cycle} finished[/dim]")
        elif kind is EventType.ERROR:
            self.errors += 1
            self.console.print(f"{prefix}[red]✗ {message}[/red]")
        elif kind is EventType.DONE:
            self.stopped = self.stopped or bool(data.get("stop_requested"))
            self.console.print(f"\n[bold green]✔ {message}[/bold green]")
        else:
            self.console.print(f"{prefix}[dim]{message}[/dim]")

    def _task_label(self, data: dict[str, Any], message: str) -> str:
        """Return ``id description`` for a task event."""
        raw_task = data.get("task")
        task = raw_task if isinstance(raw_task, dict) else {}
        return _oneline(f"{task.get('id', '?')} {task.get('description') or message}")


# ------------------------------------------------------------------ interrupts


@contextlib.contextmanager
def _interruptible(orchestrator: Orchestrator, renderer: EventRenderer) -> Iterator[None]:
    """Make the first SIGINT a graceful stop and the second an immediate abort.

    ``signal.signal`` only exists on the main thread of platforms that support
    it; everywhere else the context manager is a no-op and the default
    ``KeyboardInterrupt`` still reaches the caller.
    """
    previous: Any = None

    def _handler(signum: int, frame: FrameType | None) -> None:
        if renderer.interrupted:
            console.print("[red]Second interrupt: aborting now.[/red]")
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        renderer.interrupted = True
        renderer.stopped = True
        console.print("\n[yellow]Interrupt received: stopping after the current step.[/yellow]")
        console.print("[dim]Press Ctrl+C again to abort immediately.[/dim]")
        orchestrator.request_stop()

    try:
        previous = signal.signal(signal.SIGINT, _handler)
    except (AttributeError, NotImplementedError, OSError, RuntimeError, ValueError):
        yield
        return

    try:
        yield
    finally:
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            signal.signal(signal.SIGINT, previous)


# --------------------------------------------------------------------- helpers


@lru_cache(maxsize=1)
def _package_version() -> str:
    """Return the installed distribution version, or the source fallback."""
    try:
        return metadata_version("furrow")
    except PackageNotFoundError:
        return FALLBACK_VERSION


def _counts(state: dict[str, Any]) -> dict[str, int]:
    """Count completed, failed, and outstanding tasks in an orchestrator state."""
    tasks = [task for task in state.get("tasks") or [] if isinstance(task, dict)]
    completed = sum(1 for task in tasks if task.get("status") == COMPLETED)
    failed = sum(1 for task in tasks if task.get("status") == FAILED)
    return {
        "total": len(tasks),
        "completed": completed,
        "failed": failed,
        "outstanding": len(tasks) - completed - failed,
    }


def _report(orchestrator: Orchestrator, renderer: EventRenderer) -> int:
    """Print the closing summary from ``orchestrator.state()`` and pick an exit code."""
    state = orchestrator.state()
    counts = _counts(state)
    done = bool(state.get("done"))

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column(overflow="fold")
    grid.add_row("goal", str(state.get("goal") or ""))
    grid.add_row("cycles", str(state.get("cycles", 0)))
    grid.add_row("tasks completed", str(counts["completed"]))
    grid.add_row("tasks failed", str(counts["failed"]))
    if counts["outstanding"]:
        grid.add_row("tasks outstanding", str(counts["outstanding"]))
    if renderer.errors:
        grid.add_row("errors", str(renderer.errors))
    console.print(Panel(grid, title="Summary", border_style="green" if done else "yellow"))

    if done:
        console.print("[bold green]Goal complete.[/bold green]")
        return EXIT_OK
    if renderer.stopped:
        console.print("[bold yellow]Stopped on request before the goal was met.[/bold yellow]")
        return EXIT_OK
    console.print("[bold red]Goal not met: the loop halted anyway.[/bold red]")
    return EXIT_ERROR


def _drive(orchestrator: Orchestrator, max_cycles: int | None, renderer: EventRenderer) -> int:
    """Run the orchestrator and translate the outcome into a process exit code."""
    try:
        with _interruptible(orchestrator, renderer):
            asyncio.run(orchestrator.run(max_cycles=max_cycles))
    except KeyboardInterrupt:
        orchestrator.request_stop()
        console.print("\n[red]Aborted before the loop could finish.[/red]")
        return EXIT_INTERRUPTED
    except Exception as exc:  # noqa: BLE001 - any failure becomes a reported exit code
        console.print(f"[red]Furrow run failed: {type(exc).__name__}: {exc}[/red]")
        return EXIT_ERROR
    return _report(orchestrator, renderer)


async def _plan_once(goal: str, settings: Settings) -> Plan:
    """Ask the planner for one plan; the workspace is never touched."""
    from furrow.agents.planner import PlannerAgent

    planner = PlannerAgent(settings=settings)
    params: Mapping[str, inspect.Parameter]
    try:
        params = inspect.signature(planner.plan).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {"context": ""} if "context" in params else {}
    return await planner.plan(goal, **kwargs)


def _print_plan(plan: Plan) -> None:
    """Pretty-print a plan as a task table plus its rationale."""
    table = Table(title="Plan", title_style="bold cyan", header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("description", overflow="fold")
    table.add_column("files", style="dim", overflow="fold")
    table.add_column("deps", style="dim", overflow="fold")
    for task in plan.tasks:
        table.add_row(
            task.id,
            task.description,
            ", ".join(task.files) or "-",
            ", ".join(task.dependencies) or "-",
        )
    console.print(table)
    if plan.rationale:
        console.print(Panel(plan.rationale, title="Rationale", border_style="blue"))
    if not plan.tasks:
        console.print("[yellow]The planner returned no tasks.[/yellow]")
    console.print(f"[dim]{len(plan.tasks)} task(s) planned. Nothing was executed.[/dim]")


# --------------------------------------------------------------------- commands


@click.group()
def main() -> None:
    """Furrow: an autonomous plan, execute, and test development loop."""


@main.command()
@click.argument("goal", required=False)
@_settings_options
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    max_parallel: int | None,
    test_command: str | None,
    test_timeout: float | None,
) -> None:
    """Run the development loop until the goal is met or you stop it."""
    goal = _resolve_goal(goal)
    settings = _build_settings(
        model=model,
        provider=provider,
        workspace=workspace,
        max_cycles=max_cycles,
        max_parallel=max_parallel,
        test_command=test_command,
        test_timeout=test_timeout,
    )
    renderer = EventRenderer(console)
    orchestrator = Orchestrator(goal=goal, settings=settings, on_event=renderer)
    console.print(_describe(settings))
    sys.exit(_drive(orchestrator, max_cycles, renderer))


@main.command()
@click.argument("goal", required=False)
@_settings_options
def plan(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    max_parallel: int | None,
    test_command: str | None,
    test_timeout: float | None,
) -> None:
    """Show the plan for a goal without executing anything."""
    goal = _resolve_goal(goal)
    settings = _build_settings(
        model=model,
        provider=provider,
        workspace=workspace,
        max_cycles=max_cycles,
        max_parallel=max_parallel,
        test_command=test_command,
        test_timeout=test_timeout,
    )
    console.print(_describe(settings))
    console.print("[dim]Planning...[/dim]")
    try:
        result = asyncio.run(_plan_once(goal, settings))
    except KeyboardInterrupt:
        console.print("[yellow]Planning cancelled.[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as exc:  # noqa: BLE001 - any failure becomes a reported exit code
        console.print(f"[red]Planning failed: {type(exc).__name__}: {exc}[/red]")
        sys.exit(EXIT_ERROR)
    _print_plan(result)


@main.command()
@click.option(
    "--host",
    default=DEFAULT_HOST,
    show_default=True,
    help="Address to bind; 0.0.0.0 (the default) exposes every interface.",
)
@click.option(
    "--port",
    default=DEFAULT_PORT,
    show_default=True,
    type=click.IntRange(1, 65535),
    help="TCP port to listen on.",
)
def web(host: str, port: int) -> None:
    """Serve the web console; it binds 0.0.0.0 (all interfaces) by default."""
    from furrow.web.server import run

    try:
        run(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow console stopped.[/yellow]")


@main.command()
def version() -> None:
    """Show the Furrow version and the Python it runs on."""
    console.print(f"furrow {_package_version()} (python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()