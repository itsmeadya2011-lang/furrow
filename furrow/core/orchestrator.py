from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        # State carried across cycles for completion tracking and reporting.
        self._tasks: list[TaskModel] = []
        self._test_passed: bool = False
        self._semaphore: asyncio.Semaphore | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═ Cycle {self.cycles} ═[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if (
                self.client.settings.max_cycles > 0
                and self.cycles >= self.client.settings.max_cycles
            ):
                console.print(
                    f"[yellow]Reached max_cycles limit ({self.client.settings.max_cycles}). Stopping.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._tasks = list(plan.tasks)

        if not self._tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._test_passed = True
            return

        concurrency = max(self.client.settings.max_parallel_tasks or 1, 1)
        self._semaphore = asyncio.Semaphore(concurrency)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [self._execute_task(task) for task in self._tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(self._tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed:[/red] {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)
        self._test_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_task(self, task: TaskModel) -> str:
        assert self._semaphore is not None
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed < len(tasks):
            return False
        # All tasks completed: only halt if the latest test run passed.
        return self._test_passed

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
