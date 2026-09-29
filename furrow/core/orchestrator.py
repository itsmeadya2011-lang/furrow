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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            plan = await self._cycle()
            if not plan.tasks:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self._max_cycles_exceeded():
                limit = self.client.settings.max_cycles
                console.print(f"[yellow]Max cycles limit ({limit}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> Plan:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Merge the newly planned tasks into the tracked list, de-duplicating by id.
        self._merge_tasks(plan.tasks)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return plan

        # Execute the tracked entries so all_tasks stays in sync with results.
        tracked_by_id = {t.id: t for t in self.all_tasks}
        executing = [tracked_by_id[task.id] for task in plan.tasks]

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in executing
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(executing, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, executing)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return plan

    def _merge_tasks(self, tasks: list[TaskModel]) -> None:
        """Merge planned tasks into ``all_tasks`` without duplicating ids."""
        existing = {t.id: t for t in self.all_tasks}
        for task in tasks:
            if task.id in existing:
                tracked = existing[task.id]
                tracked.description = task.description
                tracked.files = task.files
                tracked.dependencies = task.dependencies
            else:
                self.all_tasks.append(task)
                existing[task.id] = task

    def _max_cycles_exceeded(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        return max_cycles > 0 and self.cycles >= max_cycles

    def _is_done(self) -> bool:
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
