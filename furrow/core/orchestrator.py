from __future__ import annotations

import asyncio
import io
from collections.abc import Awaitable, Callable

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


def _strip_rich(text: str) -> str:
    """Render a string that may contain Rich markup to plain text."""
    stripper = Console(file=io.StringIO(), no_color=True, highlight=False)
    stripper.print(text)
    return stripper.file.getvalue().rstrip("\n")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_message: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        # Tasks scheduled across all cycles, persisted for status tracking.
        self.all_tasks: list[TaskModel] = []
        self.completed = False
        self.on_message = on_message

    async def _emit(self, message: str) -> None:
        """Print a message to the console and forward it to any listener."""
        console.print(message)
        if self.on_message is not None:
            await self.on_message(_strip_rich(message))

    async def run(self) -> None:
        await self._emit(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        max_cycles = self.client.settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit("[yellow]Reached max_cycles limit. Stopping.[/yellow]")
                break

            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            stop = await self._cycle()
            if stop:
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                self.completed = True
                break

    async def _cycle(self) -> bool:
        """Run a single planning -> execution -> testing cycle.

        Returns True if the goal is considered complete and the loop should stop.
        """
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return True

        self.all_tasks.extend(plan.tasks)

        limit = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(limit) if limit and limit > 0 else None

        async def _run_task(task: TaskModel) -> str:
            if semaphore is not None:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()
            return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(_run_task(t) for t in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            return True

        await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            await self._emit(f"  - {failure}")
        await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
        self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
        return False

    def _is_done(self) -> bool:
        """True when every scheduled task completed without failures."""
        if not self.all_tasks:
            return False
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)
