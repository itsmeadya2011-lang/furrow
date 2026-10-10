from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

OutputCallback = Union[Callable[[str], None], Callable[[str], Awaitable[None]]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: Optional[LLMClient] = None,
        on_output: Optional[OutputCallback] = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.settings = Settings()
        self.on_output = on_output

    async def _emit(self, message: str) -> None:
        """Emit a message to the console and optionally to the output callback."""
        console.print(message)
        if self.on_output is not None:
            result = self.on_output(message)
            if inspect.isawaitable(result):
                await result

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            max_cycles = self.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit(
                    f"[bold yellow]Reached max_cycles limit ({max_cycles}). Halting.[/bold yellow]"
                )
                break

            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store new tasks in self.tasks, avoiding duplicates by id.
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)
                existing_ids.add(task.id)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            task_coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*task_coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        # Print cycle summary
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        pending = sum(1 for t in self.tasks if t.status == "pending")
        await self._emit(
            Panel(
                f"[green]Completed: {completed}[/green]  "
                f"[red]Failed: {failed}[/red]  "
                f"[yellow]Pending: {pending}[/yellow]  "
                f"[cyan]Total: {len(self.tasks)}[/cyan]",
                title="Cycle Summary",
                border_style="magenta",
            )
        )

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks