from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client, settings=settings)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._on_message: Callable[[str], None] | None = None

    def set_message_handler(self, handler: Callable[[str], None] | None) -> None:
        self._on_message = handler

    async def _send(self, message: str) -> None:
        console.print(message)
        if self._on_message:
            try:
                self._on_message(message)
            except Exception:
                pass

    async def run(self) -> None:
        await self._send(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._send(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self.settings and self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                await self._send(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]")
                break
            await self._cycle()
            if self._is_done():
                await self._send("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._current_plan = plan
        await self._send(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._send("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._send("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in plan.tasks
        ]
        max_parallel = self.settings.max_parallel_tasks if self.settings else 5
        batches = [tasks[i:i + max_parallel] for i in range(0, len(tasks), max_parallel)]
        results: list[BaseException | str] = []
        for batch in batches:
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._send(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("Task %s failed: %s", task.id, result)
            else:
                task.status = "completed"
                task.result = result
                await self._send(f"[green]Task {task.id} completed[/green]")
                logger.info("Task %s completed", task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._send(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed: %s", test_result.summary)
        else:
            await self._send(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._send(f"  • {failure}")
            await self._send("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            logger.warning("Tests failed. Will retry in next cycle. Failures: %s", test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
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
