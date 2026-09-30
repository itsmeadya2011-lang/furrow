from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, settings
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger("furrow")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        reporter: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self.reporter = reporter

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if settings.max_cycles and self.cycles >= settings.max_cycles:
                logger.info("Reached max_cycles=%s, stopping", settings.max_cycles)
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._report(f"Cycle {self.cycles} starting")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._report(f"Goal complete after {self.cycles} cycle(s)")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._report(f"Plan received with {len(plan.tasks)} task(s)")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
        assert self._semaphore is not None
        semaphore = self._semaphore

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(_run_task(t) for t in plan.tasks), return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._report(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._report(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._report(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._report(f"Tests failed: {test_result.summary}")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self._get_tasks())

    def _get_tasks(self) -> list[Any]:
        return self.current_plan.tasks if self.current_plan else []

    async def _report(self, msg: str) -> None:
        if self.reporter is None:
            return
        try:
            await self.reporter(msg)
        except Exception:
            logger.debug("Reporter failed", exc_info=True)
