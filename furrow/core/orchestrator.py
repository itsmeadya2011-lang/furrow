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
from furrow.config import Plan, Settings, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        orchestrator_settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.settings = orchestrator_settings or settings
        self.cycles = 0
        self.all_tasks: dict[str, Any] = {}  # task_id -> task (persisted across cycles)
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        logger.info("starting_furrow", goal=self.goal, max_cycles=self.settings.max_cycles)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                logger.warning("max_cycles_reached", max_cycles=self.settings.max_cycles)
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("goal_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.info("planning", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("plan_created", num_tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing ones, preserving completed/failed status
        self._merge_tasks(plan.tasks)

        # Filter tasks to execute: pending tasks whose dependencies are met
        executable_tasks = self._get_executable_tasks()
        if not executable_tasks:
            logger.warning("no_executable_tasks", pending_tasks=len([t for t in self.all_tasks.values() if t.status == "pending"]))
            console.print("[yellow]No executable tasks (all pending tasks have unmet dependencies).[/yellow]")
            return

        logger.info("executing_tasks", num_tasks=len(executable_tasks), task_ids=[t.id for t in executable_tasks])

        # Limit parallel execution
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_semaphore(task):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_with_semaphore(task) for task in executable_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(executable_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        logger.info("running_tests")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, list(self.all_tasks.values()))

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "test_result": test_result.model_dump(),
            "tasks_snapshot": {tid: {"status": t.status, "result": t.result} for tid, t in self.all_tasks.items()},
        })

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _merge_tasks(self, new_tasks: list[Any]) -> None:
        """Merge new tasks from planner with existing tasks, preserving status."""
        for task in new_tasks:
            if task.id in self.all_tasks:
                # Preserve existing status if already completed/failed
                existing = self.all_tasks[task.id]
                if existing.status in ("completed", "failed"):
                    # Keep the completed task, but update description/files if changed
                    existing.description = task.description
                    existing.files = task.files
                    existing.dependencies = task.dependencies
                else:
                    # Update pending task with new info
                    existing.description = task.description
                    existing.files = task.files
                    existing.dependencies = task.dependencies
            else:
                self.all_tasks[task.id] = task

    def _get_executable_tasks(self) -> list[Any]:
        """Get pending tasks whose dependencies are all completed."""
        executable = []
        for task in self.all_tasks.values():
            if task.status != "pending":
                continue
            deps_met = all(
                self.all_tasks.get(dep_id, None) and self.all_tasks[dep_id].status == "completed"
                for dep_id in task.dependencies
            )
            if deps_met:
                executable.append(task)
        return executable

    def _get_tasks(self) -> list[Any]:
        return list(self.all_tasks.values())

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        pending = sum(1 for t in tasks if t.status == "pending")

        if failed > 0:
            return False
        if pending > 0:
            return False
        if completed >= len(tasks):
            return True
        return False