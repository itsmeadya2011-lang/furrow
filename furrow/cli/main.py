"""Command line interface for Furrow.

Commands:
    furrow start "<goal>"  start (or resume) an agent run against a goal
    furrow status          show the persisted state of the last run
    furrow resume          continue the persisted run
    furrow web             serve the web UI
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any

import click

from furrow.config import RunResult, Settings, settings
from furrow.core.events import Event, EventBus, EventType
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import RunState, StateStore
from furrow.logging import configure_logging

logger = logging.getLogger(__name__)

WORKSPACE_HELP = "Project directory the agent works in."
MAX_CYCLES_HELP = "Maximum number of cycles (0 = unlimited)."


class ProgressReporter:
    """Real-time progress reporter for Furrow runs."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        self.current_cycle = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.tasks_blocked = 0
        self.last_event_time = time.time()
    
    def on_event(self, event: Event) -> None:
        """Handle an event from the orchestrator."""
        if event.type == EventType.CYCLE_STARTED:
            self.current_cycle = event.data.get("cycle", 0)
            elapsed = time.time() - self.start_time
            if self.verbose:
                click.echo(f"\n{'='*60}")
                click.echo(f"Cycle {self.current_cycle} started (elapsed: {elapsed:.1f}s)")
                click.echo(f"{'='*60}")
            else:
                click.echo(f"\n[Cycle {self.current_cycle}] Starting... ({elapsed:.1f}s elapsed)")
        
        elif event.type == EventType.PLAN_CREATED:
            task_count = event.data.get("tasks", 0)
            rationale = event.data.get("rationale", "")
            if self.verbose:
                click.echo(f"Plan: {rationale}")
                click.echo(f"Tasks: {task_count}")
            else:
                click.echo(f"[Cycle {self.current_cycle}] Plan created: {task_count} tasks - {rationale[:80]}...")
        
        elif event.type == EventType.TASK_STARTED:
            task_id = event.data.get("task", "unknown")
            if self.verbose:
                click.echo(f"  → Task {task_id} started")
            else:
                click.echo(f"  [Task {task_id}] Running...")
        
        elif event.type == EventType.TASK_FINISHED:
            task_id = event.data.get("task", "unknown")
            success = event.data.get("success", False)
            summary = event.data.get("summary", "")
            if success:
                self.tasks_completed += 1
                if self.verbose:
                    click.echo(f"  ✓ Task {task_id} completed: {summary[:100]}")
                else:
                    click.echo(f"  [Task {task_id}] ✓ Completed")
            else:
                self.tasks_failed += 1
                if self.verbose:
                    click.echo(f"  ✗ Task {task_id} failed: {summary[:100]}")
                else:
                    click.echo(f"  [Task {task_id}] ✗ Failed")
        
        elif event.type == EventType.TEST_STARTED:
            if self.verbose:
                click.echo(f"\nRunning tests...")
            else:
                click.echo(f"[Cycle {self.current_cycle}] Running tests...")
        
        elif event.type == EventType.TEST_FINISHED:
            passed = event.data.get("passed", False)
            summary = event.data.get("summary", "")
            if passed:
                if self.verbose:
                    click.echo(f"  ✓ Tests passed: {summary}")
                else:
                    click.secho(f"  [Tests] ✓ Passed", fg="green")
            else:
                if self.verbose:
                    click.echo(f"  ✗ Tests failed: {summary}")
                else:
                    click.secho(f"  [Tests] ✗ Failed", fg="red")
        
        elif event.type == EventType.CYCLE_FINISHED:
            cycle = event.data.get("cycle", 0)
            elapsed = time.time() - self.start_time
            if self.verbose:
                click.echo(f"Cycle {cycle} finished (total elapsed: {elapsed:.1f}s)")
            else:
                click.echo(f"[Cycle {cycle}] Finished ({elapsed:.1f}s total)")
        
        elif event.type == EventType.STATE_SAVED:
            if self.verbose:
                path = event.data.get("path", "")
                click.echo(f"State saved to {path}")
        
        elif event.type == EventType.LOG:
            if self.verbose:
                click.echo(f"  LOG: {event.message}")
        
        elif event.type == EventType.ERROR:
            click.secho(f"  ERROR: {event.message}", fg="red")
    
    def print_final_summary(self, result: RunResult) -> None:
        """Print final summary with timing info."""
        elapsed = time.time() - self.start_time
        click.echo(f"\n{'='*60}")
        click.echo(f"Run completed in {elapsed:.1f}s")
        click.echo(f"{'='*60}")


def _run_settings(
    model: str | None = None,
    max_cycles: int | None = None,
    workspace: Path | None = None,
) -> Settings:
    """Return a validated copy of the global settings with the given overrides applied.

    The module level ``settings`` object is never mutated: a fresh validated
    instance is constructed for this run only.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = Path(workspace)
    merged = {**settings.model_dump(), **updates}
    return Settings.model_validate(merged)


def _result_line(passed: bool | None, skipped: bool, tests_run: int) -> tuple[str, str]:
    """Return the (verdict, colour) of the one-line result of a run."""
    if skipped:
        return ("SKIPPED - no test runner detected.", "yellow")
    if passed:
        return ("PASSED - all tests passed.", "green")
    if tests_run == 0:
        return ("NOT RUN - no tests were executed.", "yellow")
    return ("FAILED - tests did not pass.", "red")


def _print_summary(result: RunResult) -> None:
    """Print the summary of a finished run."""
    click.echo("Run summary")
    click.echo(f"  Cycles:          {result.cycles}")
    click.echo(f"  Tasks completed: {result.tasks_completed}")
    click.echo(f"  Tasks failed:    {result.tasks_failed}")
    blocked = getattr(result, "tasks_blocked", 0) or 0
    click.echo(f"  Tasks blocked:  {blocked}")
    click.echo(f"  Stopped reason:  {result.stopped_reason}")
    skipped = bool(getattr(result, "skipped", False))
    tests_run = getattr(result, "tests_run", 0) or 0
    text, colour = _result_line(result.passed, skipped, tests_run)
    click.secho(f"  Result: {text}", fg=colour)
    usage = getattr(result, "usage", None)
    if usage is not None and usage.calls > 0:
        click.echo(
            f"  LLM usage:       {usage.calls} calls, {usage.total_tokens} tokens "
            f"(in {usage.input_tokens} / out {usage.output_tokens})"
        )
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        if cost > 0:
            click.echo(f"  LLM cost:        ${cost:.6f}")
    files_changed = getattr(result, "files_changed", None) or []
    if files_changed:
        click.echo(f"  Files changed:    {', '.join(files_changed)}")
    if result.summary:
        click.echo(f"  Summary: {result.summary}")


def _print_state(state: RunState) -> None:
    """Print the persisted run state."""
    click.echo("Furrow run status")
    click.echo(f"  Original goal: {state.original_goal}")
    click.echo(f"  Current goal:  {state.goal}")
    click.echo(f"  Cycles:        {state.cycles}")
    click.echo(f"  Updated:       {state.updated_at}")
    tests_run = getattr(state, "tests_run", 0) or 0
    skipped = bool(getattr(state, "last_skipped", False))
    passed = getattr(state, "last_passed", None)
    text, colour = _result_line(passed=passed, skipped=skipped, tests_run=tests_run)
    click.echo(f"  Tests run:     {tests_run}")
    click.secho(f"  Result:        {text}", fg=colour)
    usage = getattr(state, "usage", None)
    if usage is not None and usage.calls > 0:
        click.echo(f"  LLM usage:       {usage.calls} calls, {usage.total_tokens} tokens")
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        if cost > 0:
            click.echo(f"  LLM cost:        ${cost:.6f}")
    if state.tasks:
        click.echo("Tasks:")
        for task in state.tasks:
            files_changed = getattr(task, "files_changed", None) or []
            line = f"  {task.id}  {task.status}  {task.description}"
            if files_changed:
                line += f"  [{', '.join(files_changed)}]"
            click.echo(line)
    else:
        click.echo("Tasks:  (none)")


async def _release_client(client: Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result
    is awaitable.  A failing close is logged and ignored: closing must never
    turn a successful run into a failed one.
    """
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:
        logger.warning("Could not close the LLM client: %s", exc)


async def _run_and_release(orchestrator: Orchestrator) -> RunResult:
    """Run an orchestrator and release its LLM client afterwards.

    The client is closed in a ``finally`` block, so its HTTP connections are
    released even when the run raises.
    """
    try:
        return await orchestrator.run()
    finally:
        await _release_client(getattr(orchestrator, "client", None))


async def _run_with_progress(orchestrator: Orchestrator, verbose: bool = False) -> RunResult:
    """Run an orchestrator with real-time progress reporting."""
    reporter = ProgressReporter(verbose=verbose)
    
    # Check if orchestrator has event_bus (for tests with FakeOrchestrator)
    event_bus = getattr(orchestrator, "event_bus", None)
    if event_bus is None:
        # Fallback to simple run without progress reporting
        return await _run_and_release(orchestrator)
    
    queue = event_bus.subscribe()
    
    async def process_events():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                reporter.on_event(event)
                if event.type == EventType.CYCLE_FINISHED:
                    # Check if this was the last cycle
                    if event.data.get("cycle", 0) >= orchestrator.cycles and getattr(orchestrator, "_stopped_reason", ""):
                        break
            except asyncio.TimeoutError:
                # Check if orchestrator is done
                if getattr(orchestrator, "_stopped_reason", ""):
                    break
                continue
    
    # Run orchestrator and event processor concurrently
    run_task = asyncio.create_task(_run_and_release(orchestrator))
    event_task = asyncio.create_task(process_events())
    
    try:
        result = await run_task
    finally:
        event_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await event_task
        event_bus.unsubscribe(queue)
    
    reporter.print_final_summary(result)
    return result


def _execute(goal: str, run_settings: Settings, resume: bool, verbose: bool = False) -> None:
    """Run the orchestrator and print the summary of the result."""
    store = StateStore(run_settings.state_path())
    orchestrator = Orchestrator(
        goal=goal,
        settings=run_settings,
        state_store=store,
        resume=resume,
    )
    try:
        result = asyncio.run(_run_with_progress(orchestrator, verbose=verbose))
    except KeyboardInterrupt:
        click.echo("\nFurrow stopped by user.")
        sys.exit(0)
    except Exception as exc:
        click.secho(f"Furrow failed: {exc}", fg="red")
        sys.exit(1)
    _print_summary(result)
    skipped = bool(getattr(result, "skipped", False))
    tests_run = getattr(result, "tests_run", 0) or 0
    if skipped or result.passed or tests_run == 0:
        sys.exit(0)
    else:
        sys.exit(1)


@click.group()
def main() -> None:
    """Furrow - an autonomous coding agent."""
    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the LLM model.")
@click.option("--max-cycles", type=click.IntRange(min=0), default=None, help=MAX_CYCLES_HELP)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--resume", is_flag=True, help="Resume the persisted run instead of starting fresh.")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress output.")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    workspace: Path | None,
    resume: bool,
    verbose: bool,
) -> None:
    """Start a Furrow run for GOAL (prompted for when omitted)."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    run_settings = _run_settings(model=model, max_cycles=max_cycles, workspace=workspace)
    _execute(goal=goal, run_settings=run_settings, resume=resume, verbose=verbose)


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
def status(workspace: Path | None) -> None:
    """Show the state of the persisted Furrow run."""
    run_settings = _run_settings(workspace=workspace)
    store = StateStore(run_settings.state_path())
    state = store.load()
    if state is None:
        click.echo("No furrow run state found.")
        return
    _print_state(state)


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--max-cycles", type=click.IntRange(min=0), default=None, help=MAX_CYCLES_HELP)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress output.")
def resume(workspace: Path | None, max_cycles: int | None, verbose: bool) -> None:
    """Resume the persisted Furrow run."""
    run_settings = _run_settings(max_cycles=max_cycles, workspace=workspace)
    store = StateStore(run_settings.state_path())
    state = store.load()
    if state is None:
        click.secho("No furrow run state found to resume.", fg="red")
        sys.exit(1)
    goal = state.original_goal or state.goal
    _execute(goal=goal, run_settings=run_settings, resume=True, verbose=verbose)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the web server to.")
@click.option("--port", type=int, default=8000, help="Port to bind the web server to.")
def web(host: str, port: int) -> None:
    """Run the Furrow web server."""
    from furrow.web.server import run

    try:
        run(host=host, port=port)
    except KeyboardInterrupt:
        raise
    except SystemExit as exc:
        message = str(exc.code) if exc.code is not None else "the server could not start"
        click.secho(f"Furrow failed: {message}", fg="red")
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Furrow failed: {exc}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
