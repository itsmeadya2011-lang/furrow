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
from furrow.config import Settings, TaskModel, TestResult
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
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        # All tasks ever planned, across cycles (for reporting/history).
        self.tasks: list[TaskModel] = []
        # Tasks from the most recent cycle (used for done detection).
        self.cycle_tasks: list[TaskModel] = []
        self.tests_passed = False

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self.settings.max_cycles and self.cycles > self.settings.max_cycles:
                console.print(
                    f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]"
                )
                break
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Register new tasks (avoid duplicate ids across cycles).
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)
                existing_ids.add(task.id)

        self.cycle_tasks = list(plan.tasks)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tests_passed = True
            return

        pending = [t for t in self.cycle_tasks if t.status == "pending"]
        if not pending:
            console.print("[yellow]No pending tasks in this cycle.[/yellow]")

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[
                    WorkerAgent(task=t, client=self.client, settings=self.settings).run()
                    for t in pending
                ],
                return_exceptions=True,
            )

        for task, result in zip(pending, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, self.cycle_tasks)

        self.tests_passed = test_result.passed
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.cycle_tasks:
            return False
        if any(t.status == "failed" for t in self.cycle_tasks):
            return False
        if not all(t.status == "completed" for t in self.cycle_tasks):
            return False
        return self.tests_passed

    def _get_tasks(self) -> list[Any]:
        return list(self.tasks)
