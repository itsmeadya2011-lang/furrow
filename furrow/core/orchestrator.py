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
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _cycle(self) -> None:
        logger.info("cycle.start", cycle=self.cycles, goal=self.goal)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        logger.info("plan.result", cycle=self.cycles, tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("no.tasks.planned", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        task_results = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task.failed", cycle=self.cycles, task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task.completed", cycle=self.cycles, task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
            task_results.append({"id": task.id, "status": task.status, "result": task.result})

        logger.info("task.results", cycle=self.cycles, tasks=task_results)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("test.passed", cycle=self.cycles, summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.error("test.failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

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
        return self.plan.tasks if self.plan else []
