from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging

console = Console()
logger = get_logger("orchestrator")


async def run_task_with_retry(task: TaskModel, client: LLMClient, max_retries: int = 3) -> str:
    """Run a worker task with exponential backoff retry."""
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(max_retries),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _run() -> str:
        return await WorkerAgent(task=task, client=client).run()

    return await _run()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        setup_logging()
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.on_output = on_output
        logger.info("orchestrator_initialized", goal=goal)

    async def _emit(self, message: str) -> None:
        """Emit output to callback if provided."""
        if self.on_output:
            await self.on_output(message)

    async def run(self) -> None:
        await self._emit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        logger.info("orchestrator_started", goal=self.goal)
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_completed", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        logger.info("planning_started", cycle=self.cycles)
        await self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        await self._emit(f"Plan: {plan.model_dump()}")
        logger.info("planning_completed", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned", cycle=self.cycles)
            return

        # Add new tasks to our task list
        self.all_tasks.extend(plan.tasks)

        logger.info("execution_started", cycle=self.cycles, task_count=len(plan.tasks))
        await self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            run_task_with_retry(task, self.client)
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed after retries: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        logger.info("testing_started", cycle=self.cycles)
        await self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
