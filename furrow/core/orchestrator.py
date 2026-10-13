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
from furrow.config import Plan, Settings, TaskModel, settings
from furrow.llm import LLMClient
from furrow.logging import setup_logging
from furrow.state import CycleRecord, FurrowState, StateManager

# Configure structured logging
setup_logging(settings.log_level)

logger = structlog.get_logger()
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None, resume: bool = False, model: str | None = None) -> None:
        self.goal = goal
        if model and settings:
            settings.model = model
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.state_manager = StateManager(workspace=self.settings.workspace)
        self.cycles = 0
        self.state: FurrowState | None = None
        self.resume = resume

    async def run(self) -> None:
        # Load or create state
        if self.resume:
            self.state = self.state_manager.load()
            if self.state:
                self.goal = self.state.goal
                self.cycles = self.state.current_cycle
                logger.info("Resumed from saved state", cycle=self.cycles, goal=self.goal)
            else:
                logger.warning("No state to resume, starting fresh")
                self.resume = False

        if not self.resume:
            self.state = self.state_manager.create_new(self.goal)
            logger.info("Started new Furrow session", goal=self.goal)

        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            # Check max cycles
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                logger.info("Max cycles reached", max_cycles=self.settings.max_cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("Goal completed", total_cycles=self.cycles)
                break

        # Final state save
        if self.state:
            self.state_manager.save(self.state)
            self._print_final_summary()

    def _print_final_summary(self) -> None:
        if not self.state:
            return
        summary = self.state.get_progress_summary()
        console.print("\n" + "=" * 50)
        console.print("[bold]Furrow Session Summary[/bold]")
        console.print("=" * 50)
        console.print(f"Goal: {summary['goal']}")
        console.print(f"Cycles: {summary['total_cycles']}")
        console.print(f"Tasks Planned: {summary['total_tasks_planned']}")
        console.print(f"Tasks Completed: {summary['total_tasks_completed']}")
        console.print(f"Tasks Failed: {summary['total_tasks_failed']}")
        console.print(f"Completion Rate: {summary['completion_rate']:.1%}")
        console.print("=" * 50)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("Plan created", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("No tasks planned", cycle=self.cycles)
            return

        # Execute tasks respecting dependencies
        completed_task_ids = set()
        failed_task_ids = set()
        task_results = {}

        # Build dependency graph
        task_map = {task.id: task for task in plan.tasks}
        pending_tasks = set(task_map.keys())

        while pending_tasks:
            # Find tasks whose dependencies are all satisfied
            ready_tasks = [
                task for task in plan.tasks
                if task.id in pending_tasks
                and all(dep in completed_task_ids for dep in task.dependencies)
            ]

            if not ready_tasks:
                # Circular dependency or all remaining tasks have unsatisfied deps
                # Run remaining tasks anyway
                ready_tasks = [task for task in plan.tasks if task.id in pending_tasks]
                logger.warning("Circular or unsatisfied dependencies detected, running remaining tasks", remaining=[t.id for t in ready_tasks])

            # Run ready tasks in parallel (up to max_parallel_tasks)
            for i in range(0, len(ready_tasks), self.settings.max_parallel_tasks):
                batch = ready_tasks[i:i + self.settings.max_parallel_tasks]
                with Status(f"[bold yellow]Executing {len(batch)} tasks in parallel...", console=console):
                    tasks = [
                        WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                        for task in batch
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                for task, result in zip(batch, results):
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        failed_task_ids.add(task.id)
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                        logger.error("Task failed", task_id=task.id, error=str(result))
                    else:
                        task.status = "completed"
                        task.result = result
                        completed_task_ids.add(task.id)
                        console.print(f"[green]Task {task.id} completed[/green]")
                        logger.info("Task completed", task_id=task.id)
                    task_results[task.id] = result

            # Remove completed/failed tasks from pending
            for task in ready_tasks:
                pending_tasks.discard(task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        # Collect modified files for potential fixing
        modified_files = list(set(f for task in plan.tasks for f in task.files if task.status == "completed"))

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Attempting to fix failures...[/yellow]")
            logger.warning("Tests failed", cycle=self.cycles, failures=test_result.failures)

            # Try to fix failures
            tester = TesterAgent(client=self.client, settings=self.settings)
            test_output = await tester._run_tests()
            fix_result = await tester.fix_failures(self.goal, plan.tasks, test_output, modified_files)

            if fix_result.get("fixes"):
                console.print("[green]Applied fixes:[/green]")
                for fix in fix_result["fixes"]:
                    console.print(f"  • {fix['file']}")
                    await self.client.write_file(fix["file"], fix["content"])
                logger.info("Applied fixes", fixes=[f["file"] for f in fix_result["fixes"]])
                # Re-run tests after fixes
                test_result = await tester.run(self.goal, plan.tasks)
                if test_result.passed:
                    console.print("[green]Tests now pass after fixes![/green]")
                    logger.info("Tests pass after fixes", cycle=self.cycles)
                else:
                    console.print("[red]Tests still failing after fixes[/red]")
                    logger.warning("Tests still failing after fixes", cycle=self.cycles)
            else:
                console.print("[yellow]Could not determine fix. Will retry in next cycle.[/yellow]")
                logger.warning("No fix determined", cycle=self.cycles)

        # Record cycle
        record = CycleRecord(
            cycle=self.cycles,
            goal=self.goal,
            plan=plan,
            test_result=test_result.model_dump() if test_result else None,
            completed_tasks=list(completed_task_ids),
            failed_tasks=list(failed_task_ids),
        )
        if self.state:
            self.state.add_cycle(record)
            self.state_manager.save(self.state)

    def _is_done(self) -> bool:
        if not self.state or not self.state.cycles:
            return False
        latest = self.state.cycles[-1]
        all_tasks = {t.id for t in latest.plan.tasks}
        completed = {t.id for t in latest.plan.tasks if t.status == "completed"}
        failed = {t.id for t in latest.plan.tasks if t.status == "failed"}

        # Done if all tasks completed and tests pass
        if completed == all_tasks and latest.test_result and latest.test_result.get("passed", False):
            return True
        # Not done if any failed
        if failed:
            return False
        return False
