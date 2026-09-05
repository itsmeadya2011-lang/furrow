from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, get_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._on_event = on_event

    async def _emit(self, message: str) -> None:
        if self._on_event:
            await self._on_event(message)

    async def run(self) -> None:
        await self._emit(f"Goal: {self.goal}")
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                msg = "[yellow]Reached max cycles. Halting.[/yellow]"
                console.print(msg)
                await self._emit(msg)
                break
            self.cycles += 1
            msg = f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
            console.print(msg)
            await self._emit(f"═══ Cycle {self.cycles} ═══")
            await self._cycle()
            if self._is_done():
                msg = "[bold green]Goal complete. Halting.[/bold green]"
                console.print(msg)
                await self._emit(msg)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.model_dump()}")

        if not plan.tasks:
            msg = "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            console.print(msg)
            await self._emit(msg)
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

            async def _run_with_limit(task: Any) -> str:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            tasks = [_run_with_limit(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                msg = f"[red]Task {task.id} failed: {result}[/red]"
                console.print(msg)
                await self._emit(msg)
            else:
                task.status = "completed"
                task.result = result
                msg = f"[green]Task {task.id} completed[/green]"
                console.print(msg)
                await self._emit(msg)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            msg = f"[green]Tests passed: {test_result.summary}[/green]"
            console.print(msg)
            await self._emit(msg)
        else:
            msg = f"[red]Tests failed: {test_result.summary}[/red]"
            console.print(msg)
            await self._emit(msg)
            for failure in test_result.failures:
                msg = f"  • {failure}"
                console.print(msg)
                await self._emit(msg)
            msg = "[yellow]Will attempt fix in next cycle.[/yellow]"
            console.print(msg)
            await self._emit(msg)
            self.goal = (
                f"{self.goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)
            )

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

    def _get_tasks(self) -> list[Any]:
        if self._current_plan is None:
            return []
        return self._current_plan.tasks
