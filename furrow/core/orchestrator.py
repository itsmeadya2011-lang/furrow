from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_message: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.on_message = on_message

    async def _emit(self, message: Any) -> None:
        console.print(message)
        if self.on_message:
            with console.capture() as cap:
                console.print(message, end="")
            text = cap.get()
            result = self.on_message(text)
            if asyncio.iscoroutine(result):
                await result

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal, max_cycles=self.settings.max_cycles)
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                logger.warning("max_cycles_reached", cycles=self.cycles, max_cycles=self.settings.max_cycles)
                await self._emit(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                logger.info("goal_complete", cycles=self.cycles)
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.info("planning_start", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("planning_complete", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            logger.info("no_tasks_planned", cycle=self.cycles)
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store tasks for tracking across cycles
        self.tasks = plan.tasks

        # Limit parallel execution with semaphore
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_semaphore(task: TaskModel) -> Any:
            async with semaphore:
                logger.info("task_start", cycle=self.cycles, task_id=task.id, description=task.description)
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    logger.info("task_completed", cycle=self.cycles, task_id=task.id)
                    return result
                except Exception as e:
                    logger.error("task_failed", cycle=self.cycles, task_id=task.id, error=str(e))
                    raise

        logger.info("execution_start", cycle=self.cycles, task_count=len(plan.tasks), max_parallel=self.settings.max_parallel_tasks)
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.warning("task_failed", cycle=self.cycles, task_id=task.id, error=str(result))
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_succeeded", cycle=self.cycles, task_id=task.id)
                await self._emit(f"[green]Task {task.id} completed[/green]")

        logger.info("testing_start", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        pending = sum(1 for t in tasks if t.status == "pending")

        logger.debug("is_done_check", completed=completed, failed=failed, pending=pending, total=len(tasks))

        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks