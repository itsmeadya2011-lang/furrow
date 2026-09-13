from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger()
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self._semaphore: asyncio.Semaphore | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("furrow_start", goal=self.goal)
        max_cycles = int(os.getenv("FURROW_MAX_CYCLES", "0"))
        parallel = int(os.getenv("FURROW_MAX_PARALLEL_TASKS", "5"))
        self._semaphore = asyncio.Semaphore(parallel)
        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if max_cycles > 0 and self.cycles > max_cycles:
                console.print(f"[red]Reached max cycles ({max_cycles}). Halting.[/red]")
                logger.warning("max_cycles_reached", cycle=self.cycles, max_cycles=max_cycles)
                break
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycle=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                self._run_worker(task)
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed >= len(self.tasks)

    async def _run_worker(self, task: Any) -> str:
        async with self._semaphore:
            worker = WorkerAgent(task=task, client=self.client)
            return await worker.run()
