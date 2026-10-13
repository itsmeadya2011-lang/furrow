from __future__ import annotations

import asyncio
import time
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
from furrow.config import TaskModel, TokenUsage, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        workspace: Path | None = None,
        max_cycles: int | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.workspace = workspace or settings.workspace
        self.max_cycles = max_cycles if max_cycles is not None else settings.max_cycles
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.completed_tasks: list[Any] = []
        self.failed_tasks: list[Any] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.total_token_usage = TokenUsage()
        self.start_time = time.time()

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.max_cycles}). Stopping.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            cycle_start = time.time()
            await self._cycle()
            cycle_duration = time.time() - cycle_start

            self.cycle_history.append({
                "cycle": self.cycles,
                "duration": cycle_duration,
                "tasks_planned": len(self.all_tasks) - len(self.completed_tasks) - len(self.failed_tasks),
                "tasks_completed": len(self.completed_tasks),
                "tasks_failed": len(self.failed_tasks),
            })

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        self._print_summary()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, self._get_context())

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add cycle info to tasks
        for task in plan.tasks:
            task.cycle = self.cycles
            task.attempts = 0
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

            async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
                async with semaphore:
                    task.attempts += 1
                    return task, await WorkerAgent(task=task, client=self.client).run()

            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                task.error = str(result)
                self.failed_tasks.append(task)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = str(result) if result is not None else None
                self.completed_tasks.append(task)
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
            # Preserve original goal but add fix context
            self.goal = f"{self.original_goal}\n\nPrevious cycle test failures to fix:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")

        if failed > 0:
            return False

        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks

    def _get_context(self) -> str:
        """Build context for planner including completed tasks and failures."""
        parts = [f"Original Goal: {self.original_goal}"]
        parts.append(f"Current Cycle: {self.cycles}")

        if self.completed_tasks:
            parts.append("\nCompleted Tasks:")
            for task in self.completed_tasks:
                parts.append(f"  - {task.id}: {task.description}")

        if self.failed_tasks:
            parts.append("\nFailed Tasks:")
            for task in self.failed_tasks:
                parts.append(f"  - {task.id}: {task.description} (Error: {task.error})")

        return "\n".join(parts)

    def _print_summary(self) -> None:
        elapsed = time.time() - self.start_time

        table = Table(title="Furrow Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Cycles", str(self.cycles))
        table.add_row("Total Tasks", str(len(self.all_tasks)))
        table.add_row("Completed", str(len(self.completed_tasks)))
        table.add_row("Failed", str(len(self.failed_tasks)))
        table.add_row("Duration", f"{elapsed:.1f}s")
        table.add_row("Total Tokens", str(self.total_token_usage.total_tokens))
        table.add_row("Estimated Cost", f"${self.total_token_usage.estimated_cost:.4f}")

        console.print(table)

        if self.failed_tasks:
            console.print("\n[red]Failed Tasks:[/red]")
            for task in self.failed_tasks:
                console.print(f"  • {task.id}: {task.description} - {task.error}")
