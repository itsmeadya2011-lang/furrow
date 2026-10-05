from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger("furrow.orchestrator")


class Orchestrator:
    """Runs the infinite development loop until the goal is complete or stopped."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        logger.info("orchestrator_start", goal=self.goal)

        while True:
            # Enforce max_cycles (0 means unlimited)
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                logger.info("max_cycles_reached", cycles=self.cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.original_goal)
            logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._current_tasks = []
            return

        self._current_tasks = plan.tasks

        # Execute tasks in parallel, limited by max_parallel_tasks
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> tuple[TaskModel, str | Exception]:
            async with semaphore:
                try:
                    result = await WorkerAgent(
                        task=task, client=self.client
                    ).run()
                    return task, result
                except Exception as e:  # noqa: BLE001
                    logger.error("worker_failed", task_id=task.id, error=str(e))
                    return task, e

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(_run_task(t) for t in plan.tasks), return_exceptions=True
            )

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        # Test
        with Status("[bold yellow]Testing...", console=console):
            try:
                test_result = await TesterAgent(
                    client=self.client
                ).run(self.original_goal, self._current_tasks)
            except Exception as e:  # noqa: BLE001
                logger.error("tester_error", error=str(e))
                test_result = TestResult(
                    passed=False,
                    summary=f"Tester error: {e}",
                    failures=[str(e)],
                )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            # Update the working goal to focus on fixing failures, but keep
            # the original goal for planning context.
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        # No tasks from the last plan means the goal is complete.
        if not self._current_tasks:
            return True
        # If any task failed, keep going (the next cycle will retry).
        if any(t.status == "failed" for t in self._current_tasks):
            return False
        # All tasks completed → goal is complete.
        return all(t.status == "completed" for t in self._current_tasks)
