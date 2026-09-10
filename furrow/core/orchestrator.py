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
from furrow.config import Plan, Settings, TestResult, settings

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: Any | None = None,
        orchestrator_settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = orchestrator_settings or settings
        self.client = client
        self.planner = PlannerAgent(client=client, settings=self.settings)
        self.cycles = 0
        self.plan: Plan | None = None
        self.all_tasks: list[Any] = []
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal, max_cycles=self.settings.max_cycles)

        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                logger.warning("max_cycles_reached", cycles=self.cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)

            cycle_result = await self._cycle()
            self.cycle_history.append(cycle_result)

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_completed", cycles=self.cycles)
                break

        self._print_summary()

    async def _cycle(self) -> dict[str, Any]:
        cycle_data: dict[str, Any] = {"cycle": self.cycles, "tasks": []}

        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned", cycle=self.cycles)
            return cycle_data

        # Update all_tasks with new tasks, preserving status of existing ones
        for new_task in plan.tasks:
            existing = next((t for t in self.all_tasks if t.id == new_task.id), None)
            if existing:
                # Preserve status if already completed
                if existing.status == "completed":
                    new_task.status = "completed"
                    new_task.result = existing.result
            self.all_tasks.append(new_task)

        cycle_data["plan"] = plan.model_dump()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            # Only run pending tasks
            pending_tasks = [t for t in plan.tasks if t.status == "pending"]
            if pending_tasks:
                tasks = [
                    WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    for task in pending_tasks
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for task, result in zip(pending_tasks, results):
                    task_data = {"id": task.id, "description": task.description}
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        task_data["status"] = "failed"
                        task_data["error"] = str(result)
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                        logger.error("task_failed", cycle=self.cycles, task_id=task.id, error=str(result))
                    else:
                        task.status = "completed"
                        task.result = result
                        task_data["status"] = "completed"
                        task_data["result"] = result
                        console.print(f"[green]Task {task.id} completed[/green]")
                        logger.info("task_completed", cycle=self.cycles, task_id=task.id)
                    cycle_data["tasks"].append(task_data)
            else:
                console.print("[yellow]All tasks already completed.[/yellow]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, plan.tasks
            )

        cycle_data["test_result"] = test_result.model_dump()

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            # Update goal to focus on fixing failures
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        return cycle_data

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")

        if failed > 0 and pending == 0:
            # All tasks attempted but some failed - not done, will retry
            return False

        if completed >= len(self.all_tasks) and pending == 0:
            return True

        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks

    def _print_summary(self) -> None:
        console.print("\n[bold]═══ Summary ═══[/bold]")
        console.print(f"Total cycles: {self.cycles}")
        console.print(f"Total tasks: {len(self.all_tasks)}")
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        console.print(f"  Completed: {completed}")
        console.print(f"  Failed: {failed}")
        console.print(f"  Pending: {pending}")
        if self.plan:
            console.print(f"Final rationale: {self.plan.rationale}")