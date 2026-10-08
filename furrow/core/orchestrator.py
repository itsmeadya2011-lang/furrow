from __future__ import annotations

import asyncio

import anthropic
import openai
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

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
        on_event: callable | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.settings: Settings = settings
        self._on_event = on_event

    async def _emit(self, message: str) -> None:
        console.print(message)
        if self._on_event is not None:
            await self._on_event(message)

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            max_cycles = self.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit(
                    f"[bold red]Max cycles ({max_cycles}) reached. Halting.[/bold red]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = await self._execute_tasks(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        task_map = {task.id: task for task in tasks}
        pending_deps = {task.id: set(task.dependencies) for task in tasks}
        completed = set()
        failed = set()

        while True:
            ready = [
                task for task in tasks
                if task.id not in completed and task.id not in failed
                and pending_deps[task.id].issubset(completed)
            ]

            if not ready:
                break

            async def run_task(task: TaskModel):
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    return task, result, None
                except (openai.APIError, anthropic.APIError, OSError) as e:
                    return task, None, e

            results = await asyncio.gather(*[run_task(task) for task in ready], return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                task, task_result, exc = result
                if exc is not None:
                    task.status = "failed"
                    task.result = str(exc)
                    await self._emit(f"[red]Task {task.id} failed: {exc}[/red]")
                    failed.add(task.id)
                else:
                    task.status = "completed"
                    task.result = task_result
                    await self._emit(f"[green]Task {task.id} completed[/green]")
                    completed.add(task.id)

        return list(task_map.values())

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
