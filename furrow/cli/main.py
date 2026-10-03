"""Command line entry point for Furrow.

Every command builds its own :class:`~furrow.config.Settings` instance from the
options it was given, layered over the environment defaults.  The module level
``furrow.config.settings`` singleton is never mutated: it is shared with every
other consumer in the process and clobbering it leaks one invocation's flags
into the rest of the program.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator

console = Console()

#: Exit code used for SIGINT style interrupts so shell scripts can detect them.
INTERRUPT_EXIT_CODE = 130
#: How many trailing history entries ``furrow status`` prints.
HISTORY_TAIL = 5
INTERRUPT_MESSAGE = "Furrow stopped by user."


@click.group()
@click.version_option(package_name="furrow")
def main() -> None:
    """Furrow: an autonomous coding agent with a parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider]),
    default=None,
    help="LLM provider to use",
)
@click.option("--model", default=None, help="Override the default model")
@click.option("--planner-model", default=None, help="Model used by the planner")
@click.option("--worker-model", default=None, help="Model used by workers")
@click.option("--tester-model", default=None, help="Model used by the tester")
@click.option("--max-cycles", type=int, default=None, help="0 = unlimited")
@click.option(
    "--max-parallel",
    "max_parallel_tasks",
    type=int,
    default=None,
    help="Maximum tasks executed in parallel",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory Furrow is allowed to edit",
)
@click.option("--test-command", default=None, help="Command used to run the tests")
@click.option("--quiet", is_flag=True, default=False, help="Silence the agent's progress output")
@click.option("--verbose", is_flag=True, default=False, help="Print verbose plan dumps")
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
    test_command: str | None,
    quiet: bool,
    verbose: bool,
) -> None:
    """Run Furrow until the goal is done, the tests pass, or it gives up."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    overrides: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "planner_model": planner_model,
        "worker_model": worker_model,
        "tester_model": tester_model,
        "max_cycles": max_cycles,
        "max_parallel_tasks": max_parallel_tasks,
        "workspace": workspace,
        "test_command": test_command,
    }
    # A silent console already swallows the plan panels, so --quiet wins over
    # --verbose instead of re-enabling them.
    if verbose and not quiet:
        overrides["verbose"] = True
    run_settings = Settings(**{key: value for key, value in overrides.items() if value is not None})

    orchestrator = Orchestrator(
        goal=goal,
        settings=run_settings,
        console=Console(quiet=True) if quiet else console,
    )
    try:
        state = asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print(f"\n[yellow]{INTERRUPT_MESSAGE}[/yellow]")
        raise SystemExit(INTERRUPT_EXIT_CODE) from None
    except Exception as exc:  # noqa: BLE001 - the CLI reports every failure as exit 1
        console.print(f"[red]Run failed: {escape(str(exc))}[/red]")
        raise SystemExit(1) from None

    dump = state.model_dump()
    _print_summary(dump)
    raise SystemExit(0 if dump.get("done") else 1)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind")
@click.option("--port", type=int, default=8000, show_default=True, help="Port to bind")
@click.option("--reload", is_flag=True, default=False, help="Reload on source changes (dev only)")
def web(host: str, port: int, reload: bool) -> None:
    """Serve the web UI (loopback only unless --host says otherwise)."""
    from furrow.web.server import run

    run(host=host, port=port, reload=reload)


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Workspace whose state file should be read",
)
def status(workspace: Path | None) -> None:
    """Show the most recently saved run state."""
    # Imported lazily: `--help` must not depend on the persistence layer.
    from furrow.config import settings as default_settings
    from furrow.state import StateStore

    run_settings = Settings(workspace=workspace) if workspace is not None else default_settings
    state_path = run_settings.resolved_state_path
    if not state_path.exists():
        _print_no_state(state_path)
        return

    state = StateStore(state_path).load()
    if state is None:
        _print_no_state(state_path)
        return
    _print_state(state.model_dump())


def _print_no_state(state_path: Path) -> None:
    console.print(f"[yellow]No saved state at {escape(str(state_path))}[/yellow]")


def _print_summary(state: dict[str, Any]) -> None:
    """Print the end-of-run summary shared by ``start`` and ``status``."""
    statuses = _task_statuses(state.get("tasks"))
    tests_passed, tests_failed = _test_counts(state.get("history"))

    console.print()
    console.print(f"[bold]Goal:[/bold] {escape(_text(state.get('goal')))}")
    console.print(f"[bold]Cycles run:[/bold] {state.get('cycles') or 0}")
    if statuses:
        detail = ", ".join(f"{name}={count}" for name, count in sorted(statuses.items()))
        console.print(f"[bold]Tasks:[/bold] {detail}")
    else:
        console.print("[bold]Tasks:[/bold] none")
    console.print(f"[bold]Tests:[/bold] {tests_passed} passed, {tests_failed} failed")
    halted = state.get("halted_reason")
    if halted:
        console.print(f"[bold yellow]Halted:[/bold yellow] {escape(_text(halted))}")
    console.print(f"[bold]Done:[/bold] {bool(state.get('done'))}")


def _print_state(state: dict[str, Any]) -> None:
    _print_summary(state)

    tasks = _items(state.get("tasks"))
    if tasks:
        table = Table("id", "status", "description")
        for task in tasks:
            row = _record(task)
            table.add_row(
                escape(_text(row.get("id"), "?")),
                escape(_text(row.get("status"), "?")),
                escape(_text(row.get("description"))),
            )
        console.print(table)

    history = _items(state.get("history"))
    console.print("\n[bold]Recent history[/bold]")
    if not history:
        console.print("[dim]nothing recorded yet[/dim]")
        return
    for entry in history[-HISTORY_TAIL:]:
        console.print(f"  {escape(_describe(entry))}")


def _describe(entry: Any) -> str:
    """Render one history entry as a single readable line."""
    if not isinstance(entry, dict):
        return _text(entry)
    parts: list[str] = []
    cycle = entry.get("cycle")
    if cycle:
        parts.append(f"cycle {cycle}")
    kind = entry.get("kind")
    if kind:
        parts.append(_text(kind))
    message = entry.get("message") or entry.get("summary")
    if message:
        parts.append(_text(message))
    return " ".join(parts) or "event"


def _task_statuses(tasks: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for task in _items(tasks):
        status = _record(task).get("status")
        counts[_text(status, "unknown")] += 1
    return counts


def _test_counts(history: Any) -> tuple[int, int]:
    """Count passed/failed test results, tolerating either event or state shapes."""
    passed = failed = 0
    for entry in _items(history):
        record = _record(entry)
        for candidate in (record, record.get("data"), record.get("test_result")):
            payload = _record(candidate)
            flag = payload.get("passed")
            if isinstance(flag, bool):
                if flag:
                    passed += 1
                else:
                    failed += 1
                break
    return passed, failed


def _record(value: Any) -> dict[str, Any]:
    """Best-effort mapping view of a dumped state entry."""
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:  # noqa: BLE001 - a broken entry must not break the summary
            return {}
        if isinstance(dumped, dict):
            return dumped
    return {}


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, default: str = "") -> str:
    """Stringify a value, unwrapping enums such as ``TaskStatus``."""
    if value is None:
        return default
    return str(getattr(value, "value", value))


if __name__ == "__main__":
    main()
