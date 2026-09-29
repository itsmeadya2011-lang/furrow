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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self._last_test_passed: bool = False

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            max_cycles = self.client.settings.max_cycles
            if max_cycles and self.cycles >= max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]"
                )
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            # A plan with no tasks means nothing left to do; treat as complete.
            self._last_test_passed = True
            return

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=console
        ):
            semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

            async def _run_task(task: TaskModel) -> str:
                async with semaphore:
                    return await WorkerAgent(
                        task=task, client=self.client
                    ).run()

            results = await asyncio.gather(
                *[_run_task(task) for task in plan.tasks],
                return_exceptions=True,
            )

        failed_tasks: list[TaskModel] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                failed_tasks.append(task)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            self._last_test_passed = True
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._last_test_passed = False
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Refocus the next plan on resolving failures.
            if failed_tasks:
                self.goal = (
                    "Fix failed tasks and failing tests:\n"
                    + "\n".join(t.result or t.description for t in failed_tasks)
                    + "\nFailing tests:\n"
                    + "\n".join(test_result.failures)
                )
            else:
                self.goal = (
                    "Fix failing tests:\n" + "\n".join(test_result.failures)
                )
            log.warning("cycle_incomplete", tests_passed=False)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            # Nothing planned means there is no remaining work.
            return True
        failed = [t for t in tasks if t.status == "failed"]
        if failed:
            return False
        completed = all(t.status == "completed" for t in tasks)
        # The goal is complete only when every task succeeded *and* the
        # test suite for this cycle passed.
        return completed and self._last_test_passed

    def _get_tasks(self) -> list[Any]:
        if self.plan is None:
            return []
        return list(self.plan.tasks)
