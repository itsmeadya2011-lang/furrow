from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.console = console if console is not None else globals()["console"]
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_tasks: list[Any] = []
        self._last_test_passed = False

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator.started", goal=self.goal)
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle.started", cycle=self.cycles)
            await self._cycle()
            logger.info("cycle.completed", cycle=self.cycles, done=self._is_done())
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator.completed", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        logger.info("planning.started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("planning.completed", cycle=self.cycles, task_count=len(plan.tasks))
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("planning.no_tasks", cycle=self.cycles)
            self._current_tasks = []
            self._last_test_passed = True
            return

        self._current_tasks = plan.tasks

        logger.info("execution.started", cycle=self.cycles, task_count=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task.failed", cycle=self.cycles, task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task.completed", cycle=self.cycles, task_id=task.id)

        logger.info("testing.started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._last_test_passed = test_result.passed

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests.passed", cycle=self.cycles, summary=test_result.summary)
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests.failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return True
        if not self._current_tasks:
            return True
        if not self._last_test_passed:
            return False
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._current_tasks
