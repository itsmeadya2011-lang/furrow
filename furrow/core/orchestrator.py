from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Any = None,
    ) -> None:
        from furrow.config import settings as default_settings

        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        logger.info("orchestrator_started", goal=self.goal)

        while True:
            self.cycles += 1

            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                logger.info("max_cycles_reached", cycles=self.cycles)
                break

            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan: Plan = await self.planner.plan(self.goal)
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            logger.info("no_tasks_planned")
            return

        # Store tasks so _is_done() can inspect their status
        self._tasks = plan.tasks

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=console
        ) as status:
            semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
            tasks = [
                self._run_worker_with_semaphore(semaphore, task)
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
            test_result = await TesterAgent(
                client=self.client
            ).run(self.goal, plan.tasks)
        self._last_test_result = test_result

        if test_result.passed:
            console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(
                f"[red]Tests failed: {test_result.summary}[/red]"
            )
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print(
                "[yellow]Will attempt fix in next cycle.[/yellow]"
            )
            logger.warning(
                "tests_failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )
            self.goal = "Fix failing tests:\n" + "\n".join(
                test_result.failures
            )

    async def _run_worker_with_semaphore(
        self, semaphore: asyncio.Semaphore, task: TaskModel
    ) -> str:
        async with semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            # No tasks at all means the goal is vacuously complete
            return True

        failed = sum(1 for t in tasks if t.status == "failed")
        completed = sum(1 for t in tasks if t.status == "completed")

        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
