from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger, configure_logging

configure_logging()
log = get_logger("orchestrator")

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.all_tasks: list[TaskModel] = []
        log.info("orchestrator_initialized", goal=goal)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.settings.max_cycles and self.cycles > self.settings.max_cycles:
                console.print(f"[bold yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/bold yellow]")
                log.info("max_cycles_reached", max_cycles=self.settings.max_cycles)
                break

            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            log.info("cycle_start", cycle=self.cycles)
            try:
                await self._cycle()
            except Exception as e:
                log.error("cycle_error", cycle=self.cycles, error=str(e), exc_info=True)
                console.print(f"[red]Cycle {self.cycles} error: {e}[/red]")
                # Continue to next cycle instead of crashing
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self._get_pending_tasks())
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("plan_created", cycle=self.cycles, tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            log.info("no_tasks_planned", cycle=self.cycles)
            return

        # Merge new tasks with existing ones, preserving completed/failed status
        self._merge_tasks(plan.tasks)

        # Execute tasks respecting dependencies
        await self._execute_tasks()

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            log.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        """Merge new tasks from planner with existing tasks, preserving status."""
        existing_by_id = {t.id: t for t in self.all_tasks}
        for task in new_tasks:
            if task.id in existing_by_id:
                # Preserve status and result from existing task
                existing = existing_by_id[task.id]
                task.status = existing.status
                task.result = existing.result
        self.all_tasks = new_tasks

    def _get_pending_tasks(self) -> list[TaskModel]:
        """Get tasks that are still pending or failed."""
        return [t for t in self.all_tasks if t.status in ("pending", "failed")]

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    async def _execute_tasks(self) -> None:
        """Execute tasks respecting dependencies and max parallelism."""
        pending = self._get_pending_tasks()
        if not pending:
            return

        # Build dependency graph
        task_map = {t.id: t for t in pending}
        completed_in_cycle = set()

        # Run tasks in topological order respecting dependencies
        while pending:
            # Find tasks whose dependencies are all satisfied
            ready = [
                t for t in pending
                if all(dep in completed_in_cycle or dep not in task_map for dep in t.dependencies)
            ]

            if not ready:
                # Circular dependency or missing dependency - run remaining anyway
                console.print("[yellow]Warning: Unresolvable dependencies, running remaining tasks[/yellow]")
                log.warning("unresolvable_dependencies", pending=[t.id for t in pending])
                ready = pending

            # Limit parallelism
            batch = ready[: self.settings.max_parallel_tasks]
            pending = [t for t in pending if t not in batch]

            with Status(f"[bold yellow]Executing {len(batch)} tasks in parallel...", console=console):
                tasks = [WorkerAgent(task=task, client=self.client, settings=self.settings).run() for task in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    log.error("task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    log.info("task_completed", task_id=task.id)
                completed_in_cycle.add(task.id)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)