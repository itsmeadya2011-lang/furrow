from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self, goal: str, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self._state_file: Path | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )

        # Check if we have a max_cycles limit
        max_cycles = self.settings.max_cycles
        if max_cycles > 0:
            console.print(f"[dim]Max cycles: {max_cycles}[/dim]")

        while True:
            self.cycles += 1

            # Check max cycles limit
            if max_cycles > 0 and self.cycles > max_cycles:
                console.print(
                    f"[bold yellow]Reached max cycles ({max_cycles}). Stopping.[/bold yellow]"
                )
                break

            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            # Auto-save state after each cycle
            if self._state_file:
                self.save_state(self._state_file)

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Show current task status
        self._show_task_status()

        with Status("[bold yellow]Planning...", console=console) as _:
            plan = await self.planner.plan(self.goal, self._get_project_context())

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking
        for task in plan.tasks:
            task.status = "pending"
            self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = str(result)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as _:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        # Record cycle history
        self.cycle_history.append(
            {
                "cycle": self.cycles,
                "goal": self.goal,
                "plan": plan.model_dump(),
                "test_result": test_result.model_dump(),
                "tasks_completed": sum(1 for t in plan.tasks if t.status == "completed"),
                "tasks_failed": sum(1 for t in plan.tasks if t.status == "failed"),
            }
        )

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")

        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def _get_project_context(self) -> str:
        """Get a summary of the project structure for the planner."""
        workspace = self.settings.workspace
        files = self.client.list_files(workspace)
        if not files:
            return "No existing files in workspace."

        # Limit to first 50 files for context
        files = files[:50]
        return f"Existing files ({len(files)}):\n" + "\n".join(f"  {f}" for f in files)

    def _show_task_status(self) -> None:
        """Display a table of all tasks and their status."""
        if not self.all_tasks:
            return

        table = Table(title="Task Status")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Cycle", style="dim")

        for task in self.all_tasks:
            status_style = {
                "pending": "yellow",
                "completed": "green",
                "failed": "red",
            }.get(task.status, "white")
            table.add_row(
                task.id,
                task.description[:60] + ("..." if len(task.description) > 60 else ""),
                f"[{status_style}]{task.status}[/{status_style}]",
                str(self.cycles) if task.status in ("completed", "failed") else "-",
            )

        console.print(table)

    def save_state(self, path: str | Path | None = None) -> None:
        """Save orchestrator state to a JSON file."""
        if path is None:
            path = self.settings.workspace / ".furrow_state.json"

        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "all_tasks": [task.model_dump() for task in self.all_tasks],
            "cycle_history": self.cycle_history,
        }

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2))
        console.print(f"[dim]State saved to {path}[/dim]")

    @classmethod
    def load_state(
        cls, path: str | Path, client: LLMClient | None = None, settings: Settings | None = None
    ) -> Orchestrator:
        """Load orchestrator state from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        data = json.loads(p.read_text())

        # Create orchestrator with the saved goal
        orchestrator = cls(goal=data["goal"], client=client, settings=settings)
        orchestrator.cycles = data["cycles"]
        orchestrator.all_tasks = [TaskModel(**t) for t in data["all_tasks"]]
        orchestrator.cycle_history = data["cycle_history"]

        console.print(f"[dim]State loaded from {path} (cycle {orchestrator.cycles})[/dim]")
        return orchestrator
