"""Click entry points for the Furrow CLI.

Two commands -- ``start`` (the development loop) and ``web`` (the monitoring UI) -- plus
the ``furrow`` group itself. :mod:`furrow.cli` re-exports ``main`` and
``[project.scripts]`` points at ``furrow.cli.main:main``, so importing this module *is*
the CLI entry point: anything not needed to print help stays out of module scope.

Import policy
-------------
The heavy imports -- :mod:`furrow.llm` (which drags in the ``anthropic`` and ``openai``
SDKs), :mod:`furrow.core` and :mod:`furrow.agents` -- are function-local, for exactly the
reason :mod:`furrow` refuses to re-export ``LLMClient``. Measured in this checkout with
``time.perf_counter`` around a cold import:

    import furrow                      0.20 s
    import furrow.llm                  1.58 s   (+1.4 s: anthropic, openai, httpx, tenacity)
    import furrow.core, furrow.agents  1.59 s
    import furrow.web.server           2.10 s

A module-level ``from furrow.core import Orchestrator`` would therefore make
``furrow --help``, ``furrow --version`` and every tab-completion probe pay 1.4 s before
printing a line. Binding the names at call time also keeps the commands monkeypatchable:
replacing ``furrow.core.Orchestrator`` is picked up by the next invocation because the
``from ... import ...`` runs again on every call.

Exit-code contract
------------------
0 success, 1 runtime failure (provider, planner, settings), 2 usage error (bad flag or
missing GOAL), 130 SIGINT.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypeVar

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.config import Provider, get_settings

if TYPE_CHECKING:
    from furrow.config import Settings
    from furrow.core import RunSummary
    from furrow.models import Plan

console = Console()
err_console = Console(stderr=True)
EXIT_OK, EXIT_ERROR, EXIT_INTERRUPTED, EXIT_USAGE = 0, 1, 130, 2

T = TypeVar("T")

#: Rich style per task status: `completed` green, `failed` red, `skipped` yellow, `pending`
#: dim. These are the *rendered* colours, not the requested ones -- rich has no `fgreen`
#: style (verified: `console.print("[fgreen]x[/fgreen]")` emits no SGR at all and drops the
#: tag), so bright green is spelled `green`. `running` is a fifth case that only a run
#: interrupted mid-cycle can surface; anything unlisted falls back to `dim`.
_STATUS_STYLES: Mapping[str, str] = {
    "completed": "green",
    "failed": "red",
    "skipped": "yellow",
    "pending": "dim",
    "running": "cyan",
}

#: Bind addresses that publish the agent to every network the host can reach. The web UI
#: has no authentication and the orchestrator it starts has file-write and subprocess
#: access, so all of these get a warning.
_WILDCARD_HOSTS = frozenset({"", "*", "0.0.0.0", "::", "[::]"})


def _package_version() -> str:
    """Version string for ``--version``, falling back when the dist is not installed.

    ``@click.version_option(package_name="furrow")`` is the nicer declaration -- it tracks
    whatever is actually installed -- but it resolves the distribution lazily, at
    ``--version`` time, and this checkout has no ``furrow`` dist-info in the venv
    (verified: ``importlib.metadata.distribution("furrow")`` raises
    ``PackageNotFoundError``). Click then turns ``furrow --version`` into a ``RuntimeError``
    traceback. Resolving once here keeps ``--version`` working in a source tree, an
    editable install and a wheel install alike, while still preferring real metadata when
    it is available.
    """
    try:
        return metadata.version("furrow")
    except metadata.PackageNotFoundError:
        from furrow import __version__

        return __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=_package_version(), prog_name="furrow")
def main() -> None:
    """Furrow — autonomous coding agent with an infinite parallel development loop."""


def _usage_error(message: str) -> NoReturn:
    """Print `message` to stderr and exit with :data:`EXIT_USAGE`."""
    err_console.print(f"[red]{message}[/red]")
    sys.exit(EXIT_USAGE)


def _guard(action: Callable[[], T]) -> T:
    """Run `action`, turning the failures furrow actually produces into a message + code.

    Without this, a missing credential, an unparseable planner reply or a provider outage
    all escape as a raw traceback, which is the wrong output for a CLI: the user needs to
    know which knob to turn, and a stack of pydantic/httpx frames does not tell them.
    ``KeyboardInterrupt`` maps to the shell's conventional 130.

    ``FURROW_DEBUG=1`` re-raises from the last handler so a full traceback is still
    available for bug reports.

    The ``LLMError`` import is local for the reason described in the module docstring.
    """
    from furrow.llm import LLMError

    try:
        return action()
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise SystemExit(EXIT_INTERRUPTED) from None
    except LLMError as exc:
        # furrow.llm already names the credential env vars in its own messages, and
        # transient failures are retried with backoff before they get here.
        err_console.print(f"[red]LLM error:[/red] {exc}")
        err_console.print(
            "[dim]Check the provider credential in your environment or .env, then retry. "
            "Transient provider failures are retried automatically.[/dim]"
        )
        raise SystemExit(EXIT_ERROR) from None
    except ValueError as exc:
        # Covers ensure_api_key(), the planner's parse failure and settings validation.
        err_console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(EXIT_ERROR) from None
    except Exception as exc:
        if os.environ.get("FURROW_DEBUG"):
            raise
        err_console.print(f"[red]Unexpected {type(exc).__name__}:[/red] {exc}")
        err_console.print("[dim]Re-run with FURROW_DEBUG=1 for the full traceback.[/dim]")
        raise SystemExit(EXIT_ERROR) from None


def _resolve_settings(
    *,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    test_command: str | None,
    test_timeout: int | None,
    log_level: str | None,
) -> Settings:
    """Build this invocation's :class:`~furrow.config.Settings` from flags and the environment.

    Only options the user actually passed are applied, so anything not on the command line
    keeps whatever ``FURROW_*`` / ``.env`` supplied. Raises ``SystemExit(EXIT_USAGE)`` for a
    flag value that cannot be honoured.
    """
    # Copy, never mutate. get_settings() returns a process-wide lru_cached singleton that
    # LLMClient, the planner and the orchestrator all hold by reference, so
    # `settings.model = model` reconfigures every other component in the process and, in a
    # long-lived interpreter (a test session, the web server), leaks this run's flags into
    # the next one. A deep copy hands `start` a private Settings that validation and the
    # agents may mutate freely; the singleton is never touched and no aliasing surprise is
    # possible.
    s = get_settings().model_copy(deep=True)

    # Model precedence: --model is the blunt instrument and seeds all three roles, then the
    # per-role flags are applied *afterwards* so the more specific flag wins. That ordering
    # is the whole contract: `--model X --worker-model Y` yields planner=X, worker=Y,
    # tester=X. `s.model` is set too because LLMClient.complete() falls back to it.
    if model is not None:
        s.model = model
        s.planner_model = model
        s.worker_model = model
        s.tester_model = model
    if planner_model is not None:
        s.planner_model = planner_model
    if worker_model is not None:
        s.worker_model = worker_model
    if tester_model is not None:
        s.tester_model = tester_model
    if provider is not None:
        s.provider = Provider(provider)
    if workspace is not None:
        # click.Path(file_okay=False) only enforces the file/dir distinction when
        # exists=True; by default a nonexistent path is passed straight through, so the
        # existence check has to happen here or the agents would silently work in a
        # directory that does not exist (and Settings.workspace_path() would resolve to it).
        resolved = workspace.expanduser().resolve()
        if not resolved.is_dir():
            _usage_error(f"--workspace {workspace} is not an existing directory.")
        s.workspace = resolved
    if max_cycles is not None:
        if max_cycles < 0:
            _usage_error("--max-cycles must be >= 0 (0 means run until the goal is met).")
        s.max_cycles = max_cycles
    if max_parallel_tasks is not None:
        if max_parallel_tasks < 1:
            _usage_error("--max-parallel-tasks must be >= 1.")
        s.max_parallel_tasks = max_parallel_tasks
    if test_timeout is not None:
        if test_timeout < 1:
            _usage_error("--test-timeout must be >= 1 second.")
        s.test_timeout = test_timeout
    if test_command is not None:
        s.test_command = test_command
    if log_level is not None:
        s.log_level = log_level
    return s


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Override the model for every role (planner, worker and tester).",
)
@click.option("--planner-model", default=None, help="Override only the planner model.")
@click.option("--worker-model", default=None, help="Override only the worker model.")
@click.option("--tester-model", default=None, help="Override only the tester model.")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider]),
    default=None,
    help="LLM provider to use.",
)
@click.option(
    "--workspace",
    "-w",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Project directory the agents operate on. Defaults to the current directory.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after this many cycles. 0 (default) means run until the goal is met.",
)
@click.option("--max-parallel-tasks", type=int, default=None, help="Concurrent worker cap.")
@click.option("--test-command", default=None, help="Explicit test command, e.g. 'pytest -q'.")
@click.option("--test-timeout", type=int, default=None, help="Test subprocess timeout in seconds.")
@click.option("--log-level", default=None, help="Log level.")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Plan once, print the plan, and exit without writing files or running tests.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Print the run summary as JSON instead of a table.",
)
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    test_command: str | None,
    test_timeout: int | None,
    log_level: str | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Run the development loop against GOAL until the goal is met or a limit is reached."""
    if not goal:
        err_console.print(
            "[red]Missing GOAL.[/red]\n"
            "Usage: furrow start [OPTIONS] GOAL\n"
            "  e.g. furrow start \"add a --version flag to every command\""
        )
        sys.exit(EXIT_USAGE)

    s = _resolve_settings(
        model=model,
        planner_model=planner_model,
        worker_model=worker_model,
        tester_model=tester_model,
        provider=provider,
        workspace=workspace,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
        test_command=test_command,
        test_timeout=test_timeout,
        log_level=log_level,
    )

    # Credential check before anything expensive: ensure_api_key() raises a ValueError that
    # names the env vars to set, and _guard turns it into one line instead of a traceback.
    # It is a no-op for providers that need no key.
    _guard(s.ensure_api_key)

    if dry_run:
        # No Orchestrator is constructed, so no worker can write a file and no tester can
        # run a subprocess. The planner only reads the workspace listing and calls the LLM.
        _guard(lambda: _plan_only(goal, s))
        return

    summary = _guard(lambda: _run_loop(goal, s))
    _render_summary(summary, as_json=as_json)


def _plan_only(goal: str, s: Settings) -> None:
    """Plan `goal` once, print the plan, and return. Writes nothing."""
    from furrow.agents import PlannerAgent

    # `settings=` by keyword: PlannerAgent's first positional parameter is the *client*.
    plan = asyncio.run(PlannerAgent(settings=s).plan(goal))
    _render_plan(plan)


def _run_loop(goal: str, s: Settings) -> RunSummary:
    """Run the orchestrator to completion and return its summary."""
    from furrow.core import Orchestrator

    return asyncio.run(Orchestrator(goal=goal, settings=s).run())


def _render_plan(plan: Plan) -> None:
    """Print `plan`'s tasks and rationale as a table inside a panel."""
    table = Table(title=f"Plan ({len(plan.tasks)} task(s))")
    table.add_column("id", style="bold cyan", no_wrap=True)
    table.add_column("description")
    table.add_column("files", style="dim")
    table.add_column("deps", style="dim")
    for task in plan.tasks:
        table.add_row(
            task.id,
            task.description,
            ", ".join(task.files) or "-",
            ", ".join(task.dependencies) or "-",
        )
    console.print(Panel(table, title="Plan", border_style="blue"))
    if plan.rationale:
        console.print(Panel(plan.rationale, title="Rationale", border_style="dim"))


def _render_summary(summary: RunSummary, *, as_json: bool) -> None:
    """Print the run summary as JSON, or as a counts table plus a per-task table."""
    # dict(...) so the renderer can never mutate the summary's own mapping.
    data = dict(summary.to_dict())
    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
        return

    counts = Table(show_header=False, box=None, padding=(0, 2))
    counts.add_column("metric", style="bold")
    counts.add_column("value", justify="right")
    counts.add_row("cycles", str(data.get("cycles", 0)))
    counts.add_row("completed", f"[green]{data.get('completed', 0)}[/green]")
    counts.add_row("failed", f"[red]{data.get('failed', 0)}[/red]")
    counts.add_row("skipped", f"[yellow]{data.get('skipped', 0)}[/yellow]")
    console.print(Panel(counts, title="Run summary", border_style="blue"))
    console.print(f"stopped_because: {data.get('stopped_because')}")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return
    table = Table(title=f"Tasks ({len(tasks)})")
    table.add_column("id", style="bold cyan", no_wrap=True)
    table.add_column("description")
    table.add_column("status")
    table.add_column("result", style="dim", overflow="fold")
    for raw in tasks:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status", ""))
        table.add_row(
            str(raw.get("id", "")),
            str(raw.get("description", "")),
            f"[{_STATUS_STYLES.get(status, 'dim')}]{status or 'unknown'}"
            f"[/{_STATUS_STYLES.get(status, 'dim')}]",
            str(raw.get("result") or ""),
        )
    console.print(table)


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind. 0.0.0.0 exposes an unauthenticated agent to your network.",
)
@click.option("--port", "-p", default=8000, show_default=True, type=int, help="Port to bind.")
def web(host: str, port: int) -> None:
    """Serve the web monitoring UI."""
    if not 1 <= port <= 65535:
        _usage_error(f"--port must be between 1 and 65535, got {port}.")
    if host in _WILDCARD_HOSTS:
        err_console.print(
            f"[bold yellow]Warning:[/bold yellow] binding {host or '*'} publishes furrow on "
            "every network interface. The endpoint is unauthenticated, and the agent it "
            "drives has file-write and subprocess access. Prefer 127.0.0.1, or put it "
            "behind an authenticating proxy."
        )
    # Imported here, not at module scope: furrow.web.server pulls in FastAPI and uvicorn
    # (~0.5 s on top of the SDK chain), which `furrow --help` and `furrow web --help` must
    # not pay for. The bind-at-call-time semantics also make it monkeypatchable.
    from furrow.web.server import run

    _guard(lambda: run(host=host, port=port))


if __name__ == "__main__":
    main()
