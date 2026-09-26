from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings = settings,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.max_cycles = settings.max_cycles
        self._semaphore: asyncio.Semaphore | None = None
        self.event_queue: asyncio.Queue[str] | None = None
        self.tests_passed: bool = False

    async def run(self) -> None:
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        self.event_queue = self.event_queue or asyncio.Queue()

        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self.emit(f"[yellow]Reached max_cycles ({self.max_cycles}). Stopping.[/yellow]")
                break

            self.cycles += 1
            await self.emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                await self.emit("[bold green]Goal complete. Halting.[/bold green]")
                break

        await self.emit("\n[bold]Task Summary:[/bold]")
        for task in self.all_tasks:
            color = "green" if task.status == "completed" else "red" if task.status == "failed" else "yellow"
            await self.emit(f"  [{color}]{task.id}: {task.status}[/{color}] - {task.description}")

    async def _cycle(self) -> None:
        await self.emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self.emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tests_passed = True
            return

        self.all_tasks.extend(plan.tasks)

        await self.emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        coros = [
            self._run_task_with_semaphore(task)
            for task in plan.tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                await self.emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self.emit(f"[green]Task {task.id} completed[/green]")

        await self.emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.tests_passed = test_result.passed

        if test_result.passed:
            await self.emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self.emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self.emit(f"  • {failure}")
            await self.emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_task_with_semaphore(self, task: TaskModel) -> str:
        if self._semaphore is not None:
            async with self._semaphore:
                return await WorkerAgent(task=task, client=self.client).run()
        return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        if not self.tests_passed:
            return False
        return all(t.status in ("completed", "failed") for t in self.all_tasks)

    async def emit(self, message: str) -> None:
        console.print(message)
        if self.event_queue is not None:
            await self.event_queue.put(message)
