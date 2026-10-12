from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


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
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--temperature", default=None, type=float, help="LLM temperature (0-2)")
@click.option("--resume/--no-resume", default=True, help="Resume from saved state")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    temperature: float | None,
    resume: bool,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Create config with overrides
    config = Settings()
    if model:
        config.model = model
    if provider:
        config.provider = Provider(provider)
    if max_cycles is not None:
        config.max_cycles = max_cycles
    if max_parallel is not None:
        config.max_parallel_tasks = max_parallel
    if temperature is not None:
        config.temperature = temperature

    # If not resuming, clear state file
    if not resume and config.state_file.exists():
        config.state_file.unlink()

    try:
        asyncio.run(Orchestrator(goal=goal, config=config).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def status() -> None:
    """Show current task status from saved state."""
    state_file = settings.state_file
    if not state_file.exists():
        console.print("[yellow]No saved state found.[/yellow]")
        return

    try:
        with open(state_file) as f:
            data = json.load(f)

        cycles = data.get("cycles", 0)
        goal = data.get("goal", "")
        tasks = data.get("tasks", [])

        console.print(
            Panel.fit(
                f"[bold green]Furrow Status[/bold green]\nGoal: {goal}\nCycles: {cycles}",
                title="Status",
            )
        )

        if tasks:
            table = Table(title="Tasks")
            table.add_column("ID", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Status", style="green")
            table.add_column("Files", style="blue")

            for task in tasks:
                status_color = {
                    "completed": "green",
                    "failed": "red",
                    "pending": "yellow",
                }.get(task.get("status", "pending"), "white")

                table.add_row(
                    task.get("id", ""),
                    task.get("description", "")[:60],
                    f"[{status_color}]{task.get('status', 'pending')}[/{status_color}]",
                    ", ".join(task.get("files", []))[:40],
                )

            console.print(table)

            # Summary
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            failed = sum(1 for t in tasks if t.get("status") == "failed")
            pending = sum(1 for t in tasks if t.get("status") == "pending")
            console.print(f"\nSummary: {completed} completed, {failed} failed, {pending} pending")
        else:
            console.print("[yellow]No tasks in state.[/yellow]")

    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Error reading state: {e}[/red]")


@main.command()
@click.option("--reset", is_flag=True, help="Reset configuration to defaults")
def config(reset: bool) -> None:
    """Show or reset configuration."""
    if reset:
        # Show current config
        console.print("[yellow]Current configuration:[/yellow]")
        for field_name, field_info in Settings.model_fields.items():
            default = field_info.default
            console.print(f"  {field_name}: {default}")
        return

    console.print("[bold]Current Configuration:[/bold]")
    for field_name, field_info in Settings.model_fields.items():
        value = getattr(settings, field_name)
        default = field_info.default
        if value != default:
            console.print(f"  [cyan]{field_name}[/cyan]: {value} [dim](default: {default})[/dim]")
        else:
            console.print(f"  {field_name}: {value}")


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


@main.command()
@click.argument("path", type=click.Path(exists=True))
def init(path: str) -> None:
    """Initialize a new Furrow project."""
    project_path = Path(path).resolve()
    env_file = project_path / ".env"

    if env_file.exists():
        console.print(f"[yellow].env already exists at {env_file}[/yellow]")
    else:
        env_file.write_text(
            "# Furrow Configuration\n# ANTHROPIC_API_KEY=your_key_here\n# OPENAI_API_KEY=your_key_here\n"
        )
        console.print(f"[green]Created .env at {env_file}[/green]")

    # Create .furrowignore
    ignore_file = project_path / ".furrowignore"
    if not ignore_file.exists():
        ignore_file.write_text("""# Furrow ignore patterns
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
.env
.furrow_state.json
*.log
.DS_Store
""")
        console.print(f"[green]Created .furrowignore at {ignore_file}[/green]")

    console.print("[green]Project initialized![/green]")


if __name__ == "__main__":
    main()