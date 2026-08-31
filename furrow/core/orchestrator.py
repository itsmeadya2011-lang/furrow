from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, configure_logging, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        configure_logging(settings.log_level)
        self.log = structlog.get_logger()
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            self.plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(self.plan.model_dump()), title="Plan", border_style="blue"))

        if not self.plan.tasks:
            self.log.info("no_tasks_planned")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.log.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.log.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.plan.tasks)

        if test_result.passed:
            self.log.info("tests_passed", summary=test_result.summary)
        else:
            self.log.error("tests_failed", summary=test_result.summary)
            for failure in test_result.failures:
                self.log.error("test_failure", failure=failure)
            self.log.info("will_attempt_fix")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed == len(tasks)

    def _get_tasks(self) -> list[Any]:
        if self.plan is None:
            return []
        return self.plan.tasks
