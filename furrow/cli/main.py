from __future__ import annotations

import asyncio
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.config import Provider, Settings, reload_settings
from furrow.core.orchestrator import Orchestrator
from furrow.events import EventBus, EventType
from furrow.llm import LLMClient, LLMConfigurationError, LLMError

console = Console()


def _print_llm_error(exc: LLMConfigurationError) -> None:
    """Print a clean, actionable LLM configuration error message."""
    console.print(f"[red]LLM configuration error: {exc}[/red]")


def build_settings(**overrides: Any) -> Settings:
    """Return an isolated Settings with CLI overrides applied."""
    model = overrides.pop("model", None)
    planner_model = overrides.pop("planner_model", None)
    worker_model = overrides.pop("worker_model", None)
    tester_model = overrides.pop("tester_model", None)

    if model is not None:
        overrides.setdefault("model", model)
        overrides.setdefault("planner_model", model)
        overrides.setdefault("worker_model", model)
        overrides.setdefault("tester_model", model)
    if planner_model is not None:
        overrides["planner_model"] = planner_model
    if worker_model is not None:
        overrides["worker_model"] = worker_model
    if tester_model is not None:
        overrides["tester_model"] = tester_model

    workspace = overrides.get("workspace")
    if workspace is not None:
        p = Path(workspace)
        if not p.exists() or not p.is_dir():
            raise click.BadParameter(
                f"Workspace does not exist or is not a directory: {workspace}"
            )
        overrides["workspace"] = p

    return reload_settings(**overrides)


def _print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    try:
        ver = importlib.metadata.version("furrow")
    except importlib.metadata.PackageNotFoundError:
        ver = "unknown"
    console.print(f"furrow {ver}")
    ctx.exit()


@click.group()
@click.option(
    "--version",
    is_flag=True,
    help="Show version and exit",
    callback=_print_version,
    expose_value=False,
    is_eager=True,
)
def main() -> None:
    """Furrow: autonomous coding agent with infinite parallel development loop."""
    pass


async def _start_impl(
    goal: str,
    settings: Settings,
    bus: EventBus,
    no_live_output: bool,
) -> Orchestrator:
    orchestrator = Orchestrator(goal=goal, settings=settings, bus=bus)
    run_task = asyncio.ensure_future(orchestrator.run())
    queue = bus.subscribe()
    first_interrupt = False
    try:
        await _drain_events(run_task, queue, no_live_output)
        await run_task
    except KeyboardInterrupt:
        if first_interrupt:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
            raise
        first_interrupt = True
        await orchestrator.stop()
        await _drain_events(run_task, queue, no_live_output)
        try:
            await run_task
        except asyncio.CancelledError:
            pass
    finally:
        bus.unsubscribe(queue)
        while not queue.empty():
            # FIX (test suite): the trailing drain ignored `no_live_output`, so
            # `--no-live-output` still printed the whole run at the end - the flag only
            # delayed the feed instead of suppressing it.
            leftover = queue.get_nowait()
            if not no_live_output:
                _render_event(leftover)
    return orchestrator


async def _drain_events(
    run_task: asyncio.Task[Any],
    queue: asyncio.Queue[Any],
    no_live_output: bool,
) -> None:
    while not run_task.done():
        get_task = asyncio.ensure_future(queue.get())
        done, _ = await asyncio.wait(
            [run_task, get_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if get_task in done:
            event = get_task.result()
            if not no_live_output:
                _render_event(event)
        else:
            get_task.cancel()
            try:
                await get_task
            except asyncio.CancelledError:
                pass


def _render_event(event: Any) -> None:
    etype = event.type
    if etype == EventType.CYCLE_START:
        console.print(f"\n[bold cyan]=== Cycle {event.cycle} ===[/bold cyan]")
    elif etype == EventType.CYCLE_END:
        pass
    elif etype == EventType.TASK_START:
        console.print(f"[cyan]Task {event.data.get('id', '?')} started[/cyan]")
    elif etype == EventType.TASK_END:
        status = event.data.get("status", "unknown")
        task_id = event.data.get("id", "?")
        console.print(f"[green]Task {task_id} {status}[/green]")
    elif etype == EventType.TEST_RESULT:
        passed = event.data.get("passed", False)
        summary = event.data.get("summary", "")
        if passed:
            console.print(f"[green]Tests passed: {summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {summary}[/red]")
            for failure in event.data.get("failures", []):
                console.print(f"  - {failure}")
    elif etype == EventType.ERROR:
        console.print(f"[red]{event.message}[/red]")
    elif etype == EventType.LOG:
        new_tasks = event.data.get("new_tasks")
        tracked = event.data.get("tracked_tasks")
        if new_tasks is not None and tracked is not None:
            console.print(
                f"[yellow]{new_tasks} new task(s) planned; "
                f"{tracked} tracked overall[/yellow]"
            )
    elif etype == EventType.RUN_END:
        reason = event.data.get("halted_reason", "unknown")
        console.print(f"[bold]Furrow halted: {reason}[/bold]")


def _print_summary(summary: dict[str, Any]) -> None:
    # FIX (test suite): `start` hands this function `Orchestrator.snapshot()`, but the flat
    # `tasks_total` / `tasks_completed` / `tasks_failed` / `tests_passed` counters only exist
    # on `RunSummary` (which carries no `usage`). Every counter therefore fell back to its
    # default and the CLI ended a successful run with "0 total, 0 completed, 0 failed" and
    # "Tests passed: False". Both shapes are now read: the explicit keys when present,
    # otherwise derived from the snapshot's per-task records and its last test verdict.
    usage = summary.get("usage", {})
    tasks = list(summary.get("tasks") or [])
    results = list(summary.get("test_results") or [])
    console.print("\n[bold]Run Summary[/bold]")
    console.print(f"  Cycles: {summary.get('cycles', 0)}")
    tasks_total = summary.get("tasks_total", len(tasks))
    tasks_completed = summary.get(
        "tasks_completed", sum(1 for task in tasks if task.get("status") == "completed")
    )
    tasks_failed = summary.get(
        "tasks_failed", sum(1 for task in tasks if task.get("status") == "failed")
    )
    console.print(
        f"  Tasks: {tasks_total} total, "
        f"{tasks_completed} completed, {tasks_failed} failed"
    )
    tests_passed = summary.get(
        "tests_passed", bool(results) and bool(results[-1].get("passed", False))
    )
    console.print(f"  Tests passed: {tests_passed}")
    console.print(f"  Halted reason: {summary.get('halted_reason', 'unknown')}")
    console.print(
        f"  Tokens: {usage.get('input_tokens', 0)} in, "
        f"{usage.get('output_tokens', 0)} out, "
        f"{usage.get('requests', 0)} requests"
    )


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all agents")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider]),
    default=None,
    help="LLM provider (anthropic, openai, ollama)",
)
@click.option(
    "--workspace", type=click.Path(path_type=str), default=None,
    help="Workspace directory",
)
@click.option("--max-cycles", type=int, default=None, help="Max cycles (0 = unlimited)")
@click.option("--max-parallel", type=int, default=None, help="Max parallel tasks")
@click.option(
    "--max-tool-iterations", type=int, default=None,
    help="Max tool iterations per task",
)
@click.option("--log-level", default=None, help="Log level")
@click.option(
    "--state-file", type=click.Path(path_type=str), default=None,
    help="State file path",
)
@click.option("--no-live-output", is_flag=True, help="Suppress live event feed")
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    workspace: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    max_tool_iterations: int | None,
    log_level: str | None,
    state_file: str | None,
    no_live_output: bool,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    overrides: dict[str, Any] = {}
    if model is not None:
        overrides["model"] = model
    if planner_model is not None:
        overrides["planner_model"] = planner_model
    if worker_model is not None:
        overrides["worker_model"] = worker_model
    if tester_model is not None:
        overrides["tester_model"] = tester_model
    if provider is not None:
        overrides["provider"] = Provider(provider)
    if workspace is not None:
        overrides["workspace"] = workspace
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel
    if max_tool_iterations is not None:
        overrides["max_tool_iterations"] = max_tool_iterations
    if log_level is not None:
        overrides["log_level"] = log_level
    if state_file is not None:
        overrides["state_file"] = state_file

    try:
        settings = build_settings(**overrides)
    except click.BadParameter as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    bus = EventBus(settings.max_event_history)
    try:
        orchestrator = asyncio.run(
            _start_impl(goal, settings, bus, no_live_output)
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except LLMConfigurationError as exc:
        _print_llm_error(exc)
        sys.exit(1)
    except LLMError as exc:
        console.print(f"[red]LLM error: {exc}[/red]")
        sys.exit(1)
    except Exception:
        console.print_exception()
        sys.exit(1)

    _print_summary(orchestrator.snapshot())
    sys.exit(0)


@main.command()
@click.option("--host", default=None, help="Host to bind")
@click.option("--port", type=int, default=None, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def web(host: str | None, port: int | None, reload: bool) -> None:
    from furrow.config import get_settings

    settings = get_settings()
    actual_host = host or settings.web_host
    actual_port = port or settings.web_port
    if reload:
        import uvicorn

        from furrow.web.server import app

        uvicorn.run(app, host=actual_host, port=actual_port, reload=True)
    else:
        from furrow.web.server import run

        run(host=actual_host, port=actual_port)


@main.command()
@click.argument("goal", required=True)
def plan(goal: str) -> None:
    settings = reload_settings()
    client = LLMClient(settings)
    planner = PlannerAgent(client=client, settings=settings)

    try:
        plan_result = asyncio.run(planner.plan(goal))
    except LLMConfigurationError as exc:
        _print_llm_error(exc)
        sys.exit(1)
    except LLMError as exc:
        console.print(f"[red]Planning failed: {exc}[/red]")
        sys.exit(1)
    except Exception:
        console.print_exception()
        sys.exit(1)

    table = Table(title=f"Plan: {goal}")
    table.add_column("ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Files", style="green")
    table.add_column("Dependencies", style="yellow")
    for task in plan_result.tasks:
        table.add_row(
            task.id,
            task.description,
            ", ".join(task.files) if task.files else "-",
            ", ".join(task.dependencies) if task.dependencies else "-",
        )
    console.print(table)
    if plan_result.rationale:
        console.print(Panel(plan_result.rationale, title="Rationale"))


if __name__ == "__main__":
    main()
