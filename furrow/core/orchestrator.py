from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.settings: Settings = self.client.settings
        self._plan: Plan | None = None
        self._completed_tasks: list[TaskModel] = []
        self._output_callback = output_callback

    async def _emit(self, message: str) -> None:
        if self._output_callback is None:
            console.print(message)
            return
        # Convert Rich renderables/objects to strings for callback
        if hasattr(message, "__rich__") or hasattr(message, "__rich_console__"):
            with console.capture() as capture:
                console.print(message)
            text = capture.get()
        else:
            text = str(message)
        result = self._output_callback(text)
        if asyncio.iscoroutine(result):
            await result

    async def run(self) -> str:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                summary = "[bold green]Goal complete. Halting.[/bold green]"
                await self._emit(summary)
                return summary
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                summary = f"[bold yellow]Reached max_cycles limit ({self.settings.max_cycles}). Halting.[/bold yellow]"
                await self._emit(summary)
                return summary

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._plan = plan
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._record_completed(task)
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _record_completed(self, task: TaskModel) -> None:
        """Accumulate a completed task, avoiding duplicates by id."""
        for existing in self._completed_tasks:
            if existing.id == task.id:
                existing.status = task.status
                existing.result = task.result
                return
        self._completed_tasks.append(task)

    def _get_tasks(self) -> list[TaskModel]:
        """Return the union of the current plan's tasks and previously completed tasks."""
        tasks: list[TaskModel] = []
        if self._plan is not None:
            tasks.extend(self._plan.tasks)
        for completed in self._completed_tasks:
            if not any(t.id == completed.id for t in tasks):
                tasks.append(completed)
        return tasks

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False