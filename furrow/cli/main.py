"""Command line entry point for furrow.

The CLI is a thin shell around :class:`~furrow.core.orchestrator.Orchestrator`: it
turns click options into a fresh :class:`~furrow.config.Settings` instance (the
global singleton is never mutated), streams events through rich, and maps the
resulting :class:`~furrow.config.RunResult` onto a process exit code.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from furrow import __version__
from furrow.config import Provider, RunResult, Settings
from furrow.config import settings as global_settings
from furrow.core.events import Event, EventCallback, EventKind, format_event
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

__all__ = ["main"]

console = Console()

PROVIDER_CHOICES = [provider.value for provider in Provider]
ROLE_MODEL_FIELDS = ("planner_model", "worker_model", "tester_model")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_STOPPED = 130


@dataclass
class RunOutcome:
    """Everything the CLI needs to know once a run has finished.

    Attributes:
        result: The orchestrator's summary of the run.
        stopped: ``True`` when the run was cut short by a SIGINT-triggered stop.
    """

    result: RunResult
    stopped: bool = False


class EventPrinter:
    """Render orchestrator events through rich, coloured by kind.

    Instances are plain sync callables, so they can be handed straight to
    ``Orchestrator(on_event=...)`` without wrapping them in a coroutine.
    """

    def __init__(self, *, goal: str = "", verbose: bool = False, quiet: bool = False) -> None:
        """Store the verbosity switches used while printing events.

        Args:
            goal: Goal of the run, shown in the summary panel when an event
                payload does not carry one.
            verbose: Also print raw ``log`` events.
            quiet: Print only failures (and the final summary panel).
        """
        self._goal = goal
        self._verbose = verbose
        self._quiet = quiet
        self._last_summary: dict[str, Any] | None = None

    def __call__(self, event: Event) -> None:
        """Print one event as a single coloured line.

        Args:
            event: The event emitted by the orchestrator. ``log`` events are
                dropped unless verbose, and quiet keeps only failures.
        """
        if event.kind is EventKind.LOG and not self._verbose:
            return
        if self._quiet and event.kind not in (EventKind.ERROR, EventKind.TASK_FAILED):
            return
        style = _style_for(event)
        console.print(f"[{style}]{escape(format_event(event))}[/{style}]")
        if event.kind is EventKind.DONE:
            result = _result_from_event(event)
            if result is not None:
                self.summary(result)

    def summary(self, result: RunResult) -> None:
        """Print the end-of-run panel, at most once per distinct result.

        The goal is ignored for de-duplication, so the panel shown when a
        ``done`` event carries the result is not repeated once ``run()``
        returns the very same result.

        Args:
            result: The orchestrator's summary of the run.
        """
        payload = result.model_dump(exclude={"goal"})
        if payload == self._last_summary:
            return
        self._last_summary = payload
        outcome = "[green]completed[/green]" if result.completed else "[red]incomplete[/red]"
        tests = "[green]passed[/green]" if result.tests_passed else "[red]failed[/red]"
        body = "\n".join(
            [
                f"Goal: {escape(result.goal or self._goal)}",
                f"Cycles: {result.cycles}",
                f"Tasks: [green]{result.tasks_completed} completed[/green]"
                f" · [red]{result.tasks_failed} failed[/red]",
                f"Tests: {tests}",
                f"Outcome: {outcome}",
                f"Reason: {escape(result.reason) or '-'}",
            ]
        )
        border = "green" if result.completed else "red"
        console.print(Panel(body, title="Run summary", border_style=border))


def _style_for(event: Event) -> str:
    """Return the rich colour used to render ``event``."""
    styles = {
        EventKind.CYCLE: "cyan",
        EventKind.PLAN: "blue",
        EventKind.TASK_START: "yellow",
        EventKind.TASK_DONE: "green",
        EventKind.TASK_SKIPPED: "yellow",
        EventKind.LOG: "dim",
        EventKind.DONE: "green",
        EventKind.ERROR: "red",
    }
    if event.kind is EventKind.TESTS:
        return "green" if _tests_passed(event) else "red"
    if event.kind is EventKind.TASK_FAILED:
        return "red"
    return styles.get(event.kind, "white")


def _tests_passed(event: Event) -> bool:
    """Best-effort read of whether a ``tests`` event reports a passing run."""
    passed = event.data.get("passed")
    if isinstance(passed, bool):
        return passed
    summary = event.data.get("summary")
    text = f"{event.message} {summary if isinstance(summary, str) else ''}".lower()
    return not ("fail" in text or "error" in text)


def _result_from_event(event: Event) -> RunResult | None:
    """Rebuild a :class:`RunResult` from a ``done`` event payload, if it carries one.

    The goal is left empty on purpose: event payloads describe the outcome, and
    the printer knows the goal it was built with.
    """
    payload = {key: value for key, value in event.data.items() if key in RunResult.model_fields}
    if not payload:
        return None
    payload.setdefault("goal", "")
    try:
        return RunResult.model_validate(payload)
    except ValueError:
        return None


def _resolve_workspace(ctx: click.Context, raw: str | Path | None) -> Path:
    """Return the workspace root to run in, defaulting to the configured one.

    Args:
        ctx: Click context, used to report a bad ``--workspace`` value.
        raw: The path given on the command line, if any.

    Raises:
        click.BadParameter: When the path is not an existing directory.
    """
    candidate = Path(raw).expanduser() if raw else Path(global_settings.workspace).expanduser()
    if not candidate.is_dir():
        raise click.BadParameter(
            f"workspace directory does not exist: {candidate}",
            ctx=ctx,
            param_hint="--workspace",
        )
    return candidate.resolve()


def _build_settings(
    *,
    provider: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int,
    max_parallel: int | None,
    workspace: Path,
    verbose: bool,
) -> Settings:
    """Build a fresh :class:`Settings` from CLI overrides.

    ``--model`` sets every role model, and any explicitly supplied role model
    (``--planner-model`` and friends) wins over it. The global settings singleton
    is only read, never mutated.

    Raises:
        click.BadParameter: When ``max_cycles`` or ``max_parallel`` is out of range.
    """
    if max_cycles < 0:
        raise click.BadParameter("must be 0 (unlimited) or greater", param_hint="--max-cycles")
    if max_parallel is not None and max_parallel < 1:
        raise click.BadParameter("must be 1 or greater", param_hint="--max-parallel")

    overrides: dict[str, Any] = {"workspace": workspace, "max_cycles": max_cycles, "verbose": verbose}
    if provider is not None:
        overrides["provider"] = Provider(provider)
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel
    if model:
        overrides.update({field: model for field in ("model", *ROLE_MODEL_FIELDS)})
    specific = {
        "planner_model": planner_model,
        "worker_model": worker_model,
        "tester_model": tester_model,
    }
    overrides.update({field: value for field, value in specific.items() if value})
    return Settings(**overrides)


async def _run(cfg: Settings, goal: str, on_event: EventCallback) -> RunOutcome:
    """Drive the orchestrator once, with a cooperative SIGINT stop.

    SIGINT is turned into ``await orchestrator.stop()`` so the run ends at the
    next safe point instead of unwinding through a traceback; platforms without
    ``loop.add_signal_handler`` fall back to :class:`KeyboardInterrupt`, handled
    by the caller. The LLM client is always closed.

    Args:
        cfg: Settings for this run, passed to both the client and orchestrator.
        goal: The goal to work on.
        on_event: Sync or async callback receiving progress events.

    Returns:
        A :class:`RunOutcome` carrying the result and whether a stop was asked for.
    """
    client = LLMClient(settings=cfg)
    orchestrator = Orchestrator(goal=goal, client=client, settings=cfg, on_event=on_event)
    loop = asyncio.get_running_loop()
    stop_requested = False
    stop_task: asyncio.Task[None] | None = None

    def request_stop() -> None:
        """Ask the orchestrator to stop at its next safe point (SIGINT handler)."""
        nonlocal stop_requested, stop_task
        stop_requested = True
        if stop_task is None or stop_task.done():
            stop_task = asyncio.ensure_future(orchestrator.stop())

    try:
        loop.add_signal_handler(signal.SIGINT, request_stop)
    except (NotImplementedError, RuntimeError, ValueError):
        pass
    try:
        result = await orchestrator.run()
    finally:
        with suppress(NotImplementedError, RuntimeError, ValueError):
            loop.remove_signal_handler(signal.SIGINT)
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
        await client.aclose()
    return RunOutcome(result=result, stopped=stop_requested)


@click.group()
@click.version_option(__version__, "--version", prog_name="furrow", message="%(prog)s %(version)s")
def main() -> None:
    """Furrow: an autonomous coding agent."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the model for every role.")
@click.option("--planner-model", default=None, help="Override the planner model.")
@click.option("--worker-model", default=None, help="Override the worker model.")
@click.option("--tester-model", default=None, help="Override the tester model.")
@click.option(
    "--provider",
    type=click.Choice(PROVIDER_CHOICES),
    default=None,
    help="LLM provider to use.",
)
@click.option("--max-cycles", type=int, default=0, show_default=True, help="Cycle limit (0 = unlimited).")
@click.option(
    "--max-parallel",
    type=int,
    default=None,
    help="Maximum tasks to run in parallel.",
)
@click.option(
    "--workspace",
    default=None,
    help="Workspace root (defaults to the configured workspace).",
)
@click.option(
    "--verbose/--quiet",
    "verbosity",
    default=None,
    help="--verbose also prints log events; --quiet prints failures only.",
)
@click.pass_context
def start(
    ctx: click.Context,
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    max_cycles: int,
    max_parallel: int | None,
    workspace: str | None,
    verbosity: bool | None,
) -> None:
    """Run furrow until GOAL is reached (prompted for when omitted).

    Exits 0 when the goal was completed, 1 otherwise, and 130 when stopped with Ctrl-C.
    """
    if goal is None:
        goal = click.prompt("Enter your goal for Furrow", default="", show_default=False)
    goal = goal.strip()
    if not goal:
        raise click.BadParameter("goal must not be empty", ctx=ctx, param_hint="GOAL")

    root = _resolve_workspace(ctx, workspace)
    cfg = _build_settings(
        provider=provider,
        model=model,
        planner_model=planner_model,
        worker_model=worker_model,
        tester_model=tester_model,
        max_cycles=max_cycles,
        max_parallel=max_parallel,
        workspace=root,
        verbose=verbosity is True,
    )
    printer = EventPrinter(goal=goal, verbose=verbosity is True, quiet=verbosity is False)

    try:
        outcome = asyncio.run(_run(cfg, goal, printer))
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(EXIT_STOPPED)
    except Exception as exc:
        console.print(f"[red]Run failed: {escape(f'{type(exc).__name__}: {exc}')}[/red]")
        sys.exit(EXIT_FAILED)

    printer.summary(outcome.result)
    if outcome.stopped:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(EXIT_STOPPED)
    sys.exit(EXIT_OK if outcome.result.completed else EXIT_FAILED)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind.")
@click.option("--port", type=int, default=8000, show_default=True, help="Port to listen on.")
def web(host: str, port: int) -> None:
    """Serve the furrow web UI."""
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()