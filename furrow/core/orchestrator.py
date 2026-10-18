from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings
from furrow.llm import LLMClient

console = Console()

EmitCallback = Callable[[str], Any]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        emit: EmitCallback | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.last_test_passed: bool | None = None
        self.emit = emit

    async def _emit(self, message: str) -> None:
        if self.emit is not None:
            try:
                if asyncio.iscoroutinefunction(self.emit):
                    await self.emit(message)
                else:
                    self.emit(message)
            except Exception:
                # Never let a streaming failure kill the run.
                pass

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        await self._emit(f"Started: {self.goal}")
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._emit(f"Max cycles ({self.settings.max_cycles}) reached. Halting.")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(f"Cycle {self.cycles}")
            await self._cycle()
            if self._is_done():
                await self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale}")
        for task in plan.tasks:
            await self._emit(f"  • {task.id}: {task.description} (files: {', '.join(task.files) or 'any'})")

        self.tasks = plan.tasks

        if not self.tasks:
            await self._emit("No tasks planned. Goal may be complete.")
            self.last_test_passed = True
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            if self.settings.max_parallel_tasks > 0:
                semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

                async def limited(task: Any) -> str:
                    async with semaphore:
                        return await WorkerAgent(task=task, client=self.client).run()

                worker_tasks = [limited(task) for task in self.tasks]
            else:
                worker_tasks = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in self.tasks
                ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"Task {task.id} completed: {result}")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        self.last_test_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            await self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                await self._emit(f"  • {failure}")
            await self._emit("Will attempt fix in next cycle.")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if self.last_test_passed is False:
            return False
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed == len(self.tasks)
