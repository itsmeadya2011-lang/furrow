from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = state_file or (settings.workspace / ".furrow_state.json")
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/dim]")
            except (json.JSONDecodeError, OSError, ValueError) as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.all_tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except (OSError, ValueError, TypeError) as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            if max_cycles > 0 and self.cycles > max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Get pending/failed tasks from previous cycles
        pending_tasks = [t for t in self.all_tasks if t.status in ("pending", "failed")]

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, pending_tasks)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        for task in plan.tasks:
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing:
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
                existing.status = "pending"
            else:
                self.all_tasks.append(task)

        # Filter tasks to execute this cycle (those with no unmet dependencies)
        tasks_to_run = self._get_runnable_tasks(plan.tasks)

        if not tasks_to_run:
            console.print("[yellow]All tasks blocked by dependencies.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            task_futures = [
                WorkerAgent(task=task, client=self.client).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*task_futures, return_exceptions=True)

        for task, result in zip(tasks_to_run, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_runnable_tasks(self, planned_tasks: list[TaskModel]) -> list[TaskModel]:
        """Return tasks that can run (no unmet dependencies)."""
        runnable = []
        completed_ids = {t.id for t in self.all_tasks if t.status == "completed"}

        for task in planned_tasks:
            if task.id in completed_ids:
                continue  # Already done
            deps_met = all(dep in completed_ids for dep in task.dependencies)
            if deps_met:
                runnable.append(task)

        return runnable

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")

        if failed > 0:
            return False
        if pending > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks