from __future__ import annotations

import asyncio
from typing import cast

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

_default_console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.settings = Settings()
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._console = console or _default_console

    async def run(self) -> None:
        self._console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self._console):
            plan = await self.planner.plan(self.goal)
        self._console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self._console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = cast(str, result)
                self._console.print(f"[green]Task {task.id} completed[/green]")
            self._record_task(task)

        with Status("[bold yellow]Testing...", console=self._console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._console.print(f"  • {failure}")
            self._console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _record_task(self, task: TaskModel) -> None:
        for i, existing in enumerate(self.tasks):
            if existing.id == task.id:
                self.tasks[i] = task
                return
        self.tasks.append(task)