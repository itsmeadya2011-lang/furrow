"""Command line interface for Furrow.

`main` is the ``furrow`` console-script entry point declared in
``pyproject.toml``; the command names and option spellings here are part of the
user-facing contract and must stay stable.

The CLI never mutates the module-level ``furrow.config.settings`` singleton: each
run builds a private copy so that an option such as ``--model`` cannot leak into
any other consumer in the same process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional

import click
from rich.console import Console

from furrow.agents.errors import AgentError
from furrow.agents.planner import PlannerAgent
from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient, LLMError

__all__ = ["main"]

console = Console()

#: Settings fields that hold credentials and must never be printed in full.
_SECRET_FIELDS = ("anthropic_api_key", "openai_api_key")

#: Default cap for a single rendered event line.
_TEXT_LIMIT = 200


def _text(value: Any, limit: int = _TEXT_LIMIT) -> str:
    """Collapse `value` onto one trimmed line of at most `limit` characters.

    Args:
        value: Any payload value, possibly `None`.
        limit: Maximum rendered length; longer values are ellipsised.

    Returns:
        A single-line string, empty when `value` is `None`.
    """
    if value is None:
        return ""
    flat = " ".join(str(value).split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 3)] + "..."


def _format_plan(data: Mapping[str, Any]) -> list[str]:
    """Render a ``plan`` event as a header line plus one line per task.

    Args:
        data: The ``plan`` event payload (``cycle``, ``plan``, ``rationale``).

    Returns:
        At least one non-empty line.
    """
    lines = ["[blue]plan[/blue] " + (_text(data.get("rationale"), 200) or "(no rationale)")]
    plan = data.get("plan")
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else plan
    if isinstance(tasks, list):
        if not tasks:
            lines.append("  [blue](no tasks planned)[/blue]")
        for task in tasks:
            if isinstance(task, Mapping):
                lines.append(f"  [blue]#{task.get('id')}[/blue] {_text(task.get('description'), 160)}")
            else:
                lines.append("  " + _text(task, 160))
    elif plan is not None:
        lines.append("  " + _text(plan, 200))
    return lines


def _format_tests(data: Mapping[str, Any]) -> list[str]:
    """Render a ``tests`` event as a verdict line plus one line per failure.

    Args:
        data: The ``tests`` event payload.

    Returns:
        At least one non-empty line.
    """
    passed = bool(data.get("passed"))
    colour = "green" if passed else "red"
    lines = [f"[{colour}]tests {'PASS' if passed else 'FAIL'}[/{colour}] {_text(data.get('summary'))}"]
    failures = data.get("failures")
    if isinstance(failures, list):
        lines.extend(f"  [red]! {_text(item, 200)}[/red]" for item in failures)
    return lines


#: Renderers keyed by event name; mirrors `furrow.web.server._FORMATTERS`.
_FORMATTERS: dict[str, Callable[[Mapping[str, Any]], list[str]]] = {
    "cycle_start": lambda d: [
        f"[bold cyan]── cycle {d.get('cycle')}[/bold cyan] {_text(d.get('goal'), 120)}"
    ],
    "plan": _format_plan,
    "task_start": lambda d: [
        f"  [cyan]→ [{d.get('task_id')}][/cyan] {_text(d.get('description'), 160)}"
    ],
    "task_complete": lambda d: [
        f"  [green]✔ [{d.get('task_id')}][/green] {_text(d.get('result'))}"
    ],
    "task_failed": lambda d: [
        f"  [red]✘ [{d.get('task_id')}][/red] {_text(d.get('error'))}"
    ],
    "task_blocked": lambda d: [
        f"  [yellow]⊘ [{d.get('task_id')}] blocked:[/yellow] {_text(d.get('reason'), 160)}"
    ],
    "tests": _format_tests,
    "cycle_end": lambda d: [
        f"[bold cyan]── cycle {d.get('cycle')} end[/bold cyan] "
        f"done={d.get('done')}"
        f"{' · ' + _text(d.get('reason'), 120) if d.get('reason') else ''}"
    ],
    "error": lambda d: [f"[red]!! {_text(d.get('message'), 300)}[/red]"],
    "halt": lambda d: [
        f"[bold green]══ halt[/bold green] · {d.get('cycles')} cycle(s) · "
        f"done={d.get('done')} · {_text(d.get('reason'), 160)}"
    ],
}


def _is_degenerate(line: str) -> bool:
    """Return `True` when `line` carries no information once markup is removed.

    Args:
        line: A rendered console line that may contain rich markup.

    Returns:
        `True` when the line has no alphanumeric content.
    """
    return not any(char.isalnum() for char in _strip_markup(line))


def format_event(name: str, data: Mapping[str, Any]) -> list[str]:
    """Render one orchestrator event as console lines.

    Args:
        name: An event name from the orchestrator contract.
        data: The event payload.

    Returns:
        One or more non-empty lines; unknown names produce a visible
        ``unhandled event`` line rather than silence.
    """
    renderer = _FORMATTERS.get(name)
    if renderer is None:
        return [f"[yellow]?? unhandled event: {name}[/yellow]"]
    lines = [line for line in renderer(data) if not _is_degenerate(line)]
    return lines or [f"[yellow]?? empty payload for event: {name}[/yellow]"]


def _strip_markup(line: str) -> str:
    """Return `line` without rich console markup, for emptiness checks only.

    Args:
        line: A line that may contain rich markup tags.

    Returns:
        The line with ``[tag]`` sequences removed.
    """
    out: list[str] = []
    depth = 0
    for char in line:
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif not depth:
            out.append(char)
    return "".join(out)


def make_reporter(json_mode: bool) -> Callable[[str, dict[str, Any]], None]:
    """Build the synchronous ``on_event`` callback used by ``furrow start``.

    Args:
        json_mode: Emit one JSON object per event on stdout instead of rich
            output, for piping into ``jq`` or a log collector.

    Returns:
        A callback that never raises and never blocks the caller.
    """

    def report(name: str, payload: dict[str, Any]) -> None:
        if json_mode:
            click.echo(json.dumps({"event": name, "data": payload}, default=str))
            return
        for line in format_event(name, payload):
            console.print(line, highlight=False)

    return report


def local_settings(
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_cycles: Optional[int] = None,
    workspace: Optional[Path] = None,
) -> Settings:
    """Build a run-scoped copy of the configured settings.

    The module-level ``settings`` singleton is never mutated: mutating it would
    leak this command's options into every other consumer in the process.

    Args:
        model: Model override; replaces ``model`` and the three per-role models
            because the agents pass the role models explicitly to the client.
        provider: Provider override, one of the :class:`Provider` values.
        max_cycles: Cycle budget override; ``0`` means unlimited.
        workspace: Workspace override, expanded to an absolute path.

    Returns:
        A new `Settings` instance for this run only.
    """
    updates: dict[str, Any] = {}
    if provider:
        updates["provider"] = Provider(provider)
    if model:
        updates.update(
            model=model, planner_model=model, worker_model=model, tester_model=model
        )
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = Path(workspace).expanduser()
    return settings.model_copy(update=updates)


def missing_key_hint(local: Settings) -> Optional[str]:
    """Return an actionable hint when no credential is configured.

    Args:
        local: The settings this run will use.

    Returns:
        A hint message, or `None` when the provider is usable (or needs no key).
    """
    provider = Provider(local.provider)
    if provider is Provider.ANTHROPIC:
        if not (local.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")):
            return (
                "No Anthropic API key found. Set FURROW_ANTHROPIC_API_KEY (or "
                "ANTHROPIC_API_KEY) in the environment or in .env."
            )
    elif provider is Provider.OPENAI:
        if not (local.openai_api_key or os.getenv("OPENAI_API_KEY")):
            return (
                "No OpenAI API key found. Set FURROW_OPENAI_API_KEY (or "
                "OPENAI_API_KEY) in the environment or in .env."
            )
    return None


def redact_secrets(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of `data` with every credential replaced by a marker.

    A raw key is never returned: this output is printed to a terminal and ends
    up in CI logs.  A key is shown as ``set (...1234)`` when it is long enough
    to have a distinctive tail, and as ``set`` otherwise.

    Args:
        data: The dumped settings mapping.

    Returns:
        A new mapping with `_SECRET_FIELDS` redacted.
    """
    redacted = dict(data)
    for name in _SECRET_FIELDS:
        if name in redacted:
            redacted[name] = _redacted(redacted[name])
    return redacted


def _redacted(value: Any) -> str:
    """Return a safe description of one credential value.

    Args:
        value: The raw credential, or `None`.

    Returns:
        ``"not set"``, ``"set"``, or ``"set (...1234)"``.
    """
    if not isinstance(value, str) or not value.strip():
        return "not set"
    if len(value) < 8:
        return "set"
    return f"set (...{value[-4:]})"


def _render_value(value: Any) -> str:
    """Render one settings value for the ``config`` command.

    Args:
        value: Any value from the dumped settings.

    Returns:
        A single-line human-readable string.
    """
    if value is None:
        return "(unset)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(empty)"
    if isinstance(value, dict):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


async def _print_plan(client: LLMClient, goal: str, json_mode: bool) -> None:
    """Run exactly one planner call and print the plan.

    Args:
        client: The client whose settings drive the planner.
        goal: The goal to plan for.
        json_mode: Print JSON instead of rich output.
    """
    plan = await PlannerAgent(client=client).plan(goal)
    if json_mode:
        click.echo(json.dumps({"event": "plan", "data": plan.model_dump(mode="json")}, default=str))
        return
    console.print(f"[bold]Plan for[/bold] [italic]{goal}[/italic]", highlight=False)
    for index, task in enumerate(plan.tasks, start=1):
        console.print(f"  {index}. [cyan]{task.id}[/cyan] {task.description}", highlight=False)
        if task.files:
            console.print(f"     files: {_text(task.files)}", highlight=False)
    if plan.rationale:
        console.print(f"[dim]rationale:[/dim] {plan.rationale}", highlight=False)


async def _run_session(
    client: LLMClient,
    goal: str,
    local: Settings,
    *,
    dry_run: bool,
    json_mode: bool,
) -> None:
    """Run the planner once or the full orchestrator loop, then close `client`.

    Args:
        client: The client owned by this run.
        goal: The goal to work on.
        local: Run-scoped settings.
        dry_run: Plan only, without executing or testing.
        json_mode: Emit JSON events instead of rich output.
    """
    try:
        if dry_run:
            await _print_plan(client, goal, json_mode)
            return
        orchestrator = Orchestrator(
            goal=goal,
            client=client,
            on_event=make_reporter(json_mode),
            settings=local,
        )
        await orchestrator.run()
    finally:
        closer = getattr(client, "aclose", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                await closer()


@click.group()
def main() -> None:
    """Furrow: an autonomous coding agent that works in parallel cycles."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the model for every role (planner, worker, tester).")
@click.option(
    "--provider",
    type=click.Choice([item.value for item in Provider], case_sensitive=False),
    default=None,
    help="LLM provider to use.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after this many cycles (0 = unlimited; defaults to the configured value).",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory Furrow may edit (default: the current directory).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run only the planner, print the plan, and exit without editing or testing.",
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Emit one JSON object per event on stdout instead of rich output.",
)
def start(
    goal: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    max_cycles: Optional[int],
    workspace: Optional[Path],
    dry_run: bool,
    json_mode: bool,
) -> None:
    """Run Furrow until the goal is met, the budget is spent, or you interrupt it.

    Args:
        goal: What Furrow should accomplish; prompted for when omitted.
        model: Model override applied to every role.
        provider: Provider override.
        max_cycles: Cycle budget override.
        workspace: Workspace override.
        dry_run: Plan only and exit.
        json_mode: Emit JSON events.

    Raises:
        SystemExit: ``0`` on Ctrl-C, ``1`` when the agent or the LLM layer fails.
    """
    if not goal:
        goal = click.prompt("Goal for Furrow")
    goal = goal.strip()
    if not goal:
        raise click.UsageError("a non-empty goal is required")

    local = local_settings(
        model=model, provider=provider, max_cycles=max_cycles, workspace=workspace
    )
    hint = missing_key_hint(local)
    if hint:
        console.print(f"[yellow]warning:[/yellow] {hint}")
    if max_cycles is not None and max_cycles < 0:
        raise click.UsageError("--max-cycles must be 0 (unlimited) or greater")

    client = LLMClient(settings=local)
    try:
        asyncio.run(_run_session(client, goal, local, dry_run=dry_run, json_mode=json_mode))
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except (AgentError, LLMError) as exc:
        console.print(f"[red]error:[/red] {exc}", highlight=False)
        raise SystemExit(1) from exc


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind. Use 0.0.0.0 to publish the unauthenticated agent "
    "to every host on your network.",
)
@click.option("--port", default=8000, show_default=True, help="TCP port to listen on.")
def web(host: str, port: int) -> None:
    """Serve the web UI, which streams one run at a time.

    Args:
        host: Interface to bind; loopback by default because the UI can edit
            files and spend API money without authentication.
        port: TCP port to listen on.
    """
    from furrow.web.server import run as run_server

    console.print(f"[green]Furrow UI[/green] on [bold]http://{host}:{port}[/bold]")
    run_server(host=host, port=port)


@main.command()
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    help="Print machine-readable JSON instead of key/value lines.",
)
def config(json_mode: bool) -> None:
    """Show the resolved configuration, with API keys redacted.

    Args:
        json_mode: Print JSON instead of key/value lines.
    """
    data = redact_secrets(settings.model_dump(mode="json"))
    if json_mode:
        click.echo(json.dumps(data, indent=2, sort_keys=True, default=str))
        return
    for name, value in sorted(data.items()):
        console.print(f"[cyan]{name}[/cyan] = {_render_value(value)}", highlight=False)


if __name__ == "__main__":
    main()
