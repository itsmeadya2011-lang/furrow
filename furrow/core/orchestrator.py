from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        message_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.message_callback = message_callback

    def _emit(self, msg: Any) -> None:
        console.print(msg)
        if self.message_callback is not None and isinstance(msg, str):
            clean_text = re.sub(r"\[/?[a-zA-Z0-9_=\- #]*\]", "", msg)
            self.message_callback(clean_text)

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._emit(f"[yellow]Reached max_cycles ({max_cycles}). Halting.[/yellow]")
                break

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                self._run_task(task, semaphore)
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        if any(t.status == "failed" for t in self.tasks):
            return False
        return (
            all(t.status == "completed" for t in self.tasks)
            and self.last_test_result is not None
            and self.last_test_result.passed
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
