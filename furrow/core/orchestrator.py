from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        # Track all tasks ever created, keyed by id
        self._tasks: dict[str, TaskModel] = {}
        self._all_completed: bool = False

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles limit ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                break

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            # No new tasks planned - check if all previously tracked tasks are done
            if self._all_tracked_tasks_done():
                console.print("[yellow]No tasks planned and all tracked tasks complete.[/yellow]")
                self._all_completed = True
            else:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks into our tracking dict
        for task in plan.tasks:
            if task.id not in self._tasks:
                self._tasks[task.id] = task
            else:
                # Preserve status from previous cycles, update description/files
                existing = self._tasks[task.id]
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies

        await self._execute_tasks(plan.tasks)

        # Retry any failed tasks once
        failed = [t for t in plan.tasks if t.status == "failed"]
        if failed:
            await self._retry_failed(failed)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, list(self._tasks.values())
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        max_parallel = max(1, self.settings.max_parallel_tasks)
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i : i + max_parallel]
            coros = [
                WorkerAgent(task=t, client=self.client, settings=self.settings).run()
                for t in batch
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")

    async def _retry_failed(self, failed_tasks: list[TaskModel]) -> None:
        """Retry failed tasks once with additional context."""
        console.print(f"[yellow]Retrying {len(failed_tasks)} failed task(s)...[/yellow]")
        for task in failed_tasks:
            task.status = "pending"
            task.result = None
        await self._execute_tasks(failed_tasks)

    def _all_tracked_tasks_done(self) -> bool:
        """Check if all tracked tasks are completed (not pending or failed)."""
        if not self._tasks:
            return False
        return all(t.status in ("completed",) for t in self._tasks.values())

    def _is_done(self) -> bool:
        if self._all_completed:
            return True
        # Done if all tracked tasks are completed and no failures remain
        if self._tasks and all(t.status == "completed" for t in self._tasks.values()):
            return True
        return False