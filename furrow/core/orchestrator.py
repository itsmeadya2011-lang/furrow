from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TestResult, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
        model: str | None = None,
    ) -> None:
        self.goal = goal
        self.model = model
        if client is None:
            if model is not None:
                from furrow.config import Settings
                client_settings = settings.model_copy(update={"model": model})
                client = LLMClient(settings=client_settings)
            else:
                client = LLMClient()
        self.client = client
        self.planner = PlannerAgent(client=self.client)
        self.console = console or Console()
        self.cycles = 0
        self.tasks: list[TaskModel] = []

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)
                existing_ids.add(task.id)

        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

        async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, str]:
            async with semaphore:
                return task, await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            running = [run_with_semaphore(t) for t in self.tasks if t.status == "pending"]
            results = await asyncio.gather(*running, return_exceptions=True)

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            for task in self.tasks:
                if task.status == "pending":
                    task.status = "completed"
                    task.result = "Tests passed, no further action needed"
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
            self.console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
            return True
        tasks = self._get_tasks()
        if not tasks:
            return False
        failed = any(t.status == "failed" for t in tasks)
        if failed:
            return False
        completed = all(t.status == "completed" for t in tasks)
        return completed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
