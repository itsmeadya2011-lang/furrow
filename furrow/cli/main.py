from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

import click
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _make_renderer() -> Callable[[dict], None]:
    def render(event: dict) -> None:
        etype = event.get("type")
        if etype == "start":
            goal = event.get("goal", "")
            console.print()
            console.print(Panel(Text("Furrow", style="bold green"), subtitle=goal, border_style="green"))
            console.print()
        elif etype == "cycle.start":
            cycle = event.get("cycle", 0)
            console.print(f"\n[cyan]═══ Cycle {cycle} ═══[/cyan]")
        elif etype == "plan":
            plan = event.get("plan", {})
            console.print(Panel(Pretty(plan), title="Plan", border_style="blue"))
        elif etype == "task.start":
            task_id = event.get("task_id", "?")
            console.print(f"[dim yellow]Task {task_id} starting...[/dim yellow]")
        elif etype == "task.complete":
            task_id = event.get("task_id", "?")
            summary = event.get("summary", "")
            if len(summary) > 200:
                summary = summary[:197] + "..."
            console.print(f"[green]Task {task_id} completed[/green] {summary}")
        elif etype == "task.failed":
            task_id = event.get("task_id", "?")
            error = event.get("error", "")
            console.print(f"[red]Task {task_id} failed: {error}[/red]")
        elif etype == "test.result":
            result = event.get("result", {})
            passed = result.get("passed", False)
            summary = result.get("summary", "")
            failures = result.get("failures", [])
            if passed:
                console.print(f"[green]Tests passed: {summary}[/green]")
            else:
                console.print(f"[red]Tests failed: {summary}[/red]")
                for failure in failures:
                    console.print(f"  [red]• {failure}[/red]")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        elif etype == "halt":
            reason = event.get("reason", "")
            cycles = event.get("cycles", 0)
            if reason in ("complete", "no_tasks"):
                console.print(f"\n[bold green]Goal complete after {cycles} cycles.[/bold green]")
            elif reason == "max_cycles":
                console.print("\n[bold yellow]Halted (max cycles reached).[/bold yellow]")
            else:
                console.print(f"\n[bold yellow]Halted: {reason} after {cycles} cycles.[/bold yellow]")
        elif etype == "error":
            stage = event.get("stage", "unknown")
            error = event.get("error", "")
            console.print(f"[red]Error in {stage}: {error}[/red]")

    return render


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider",
)
@click.option("--max-cycles", type=int, default=None, help="Maximum orchestration cycles")
@click.option("--max-parallel", type=int, default=None, help="Maximum parallel tasks")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model
    if provider:
        from furrow.config import Provider

        settings.provider = Provider(provider.upper())
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel

    orchestrator = Orchestrator(goal=goal)
    renderer = _make_renderer()
    orchestrator.event_bus.subscribe(renderer)

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()