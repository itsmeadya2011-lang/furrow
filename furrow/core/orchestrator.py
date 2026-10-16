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
from furrow.config import Settings, TaskModel
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
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.emit = emit

    async def _emit(self, message: str) -> None:
        if self.emit:
            if asyncio.iscoroutinefunction(self.emit):
                await self.emit(message)
            else:
                self.emit(message)

    async def _log(self, message: str, style: str | None = None) -> None:
        if style:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            console.print(message)
        await self._emit(message)

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        await self._emit(f"Started: {self.goal}")
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._log("[yellow]Max cycles reached. Halting.[/yellow]", "yellow")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(f"=== Cycle {self.cycles} ===")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale}")

        self.tasks = plan.tasks
        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("No tasks planned. Goal may be complete.")
            return

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
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"Task {task.id} completed: {result}")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            await self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                await self._emit(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._emit("Will attempt fix in next cycle.")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if any(t.status == "failed" for t in self.tasks):
            return False
        return all(t.status == "completed" for t in self.tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks
