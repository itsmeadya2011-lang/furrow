from __future__ import annotations

import asyncio
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from structlog import get_logger

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskStatus, TestResult, settings
from furrow.llm import LLMClient

logger = get_logger()
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._plan: Plan | None = None
        self._tasks: list[Any] = []
        self._all_tasks_completed = False

    async def run(self) -> None:
        logger.info("furrow_started", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                self.cycles += 1
                logger.info("cycle_started", cycle=self.cycles, goal=self.goal)
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    logger.info("goal_complete", cycles=self.cycles)
                    break
                if self._exceeded_max_cycles():
                    console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                    logger.info("max_cycles_reached", cycles=self.cycles, max_cycles=settings.max_cycles)
                    break
        except KeyboardInterrupt:
            logger.info("stopped_by_user", cycles=self.cycles)
            console.print("\n[yellow]Furrow stopped by user.[/yellow]")
            raise
        except Exception as e:
            logger.error("orchestrator_error", error=str(e), cycles=self.cycles)
            console.print(f"[red]Error: {e}[/red]")
            raise

    async def _cycle(self) -> None:
        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
            self._plan = plan
            self._tasks = list(plan.tasks)
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                self._all_tasks_completed = True
                return

            await self._execute_tasks(plan.tasks)
            await self._run_tests(plan.tasks)
            self._all_tasks_completed = True
        except Exception as e:
            logger.error("cycle_error", cycle=self.cycles, error=str(e))
            console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")
            raise

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks_coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in tasks
            ]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = TaskStatus.FAILED
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.info("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

    async def _run_tests(self, tasks: list[Any]) -> None:
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.info("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self._all_tasks_completed:
            return True
        if not self._tasks:
            return False
        completed = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self._tasks if t.status == TaskStatus.FAILED)
        if failed > 0:
            return False
        return completed >= len(self._tasks)

    def _exceeded_max_cycles(self) -> bool:
        return settings.max_cycles > 0 and self.cycles >= settings.max_cycles

    @property
    def plan(self) -> Plan | None:
        return self._plan

    @property
    def tasks(self) -> list[Any]:
        return self._tasks
