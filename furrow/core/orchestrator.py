from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, State, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.state = State(goal=goal)
        self.resume = resume
        if resume:
            self._load_state()

    async def run(self) -> None:
        logger.info("starting_furrow", goal=self.goal, resume=self.resume)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if settings.max_cycles > 0 and self.state.completed_cycles >= settings.max_cycles:
                logger.info("max_cycles_reached", max_cycles=settings.max_cycles)
                console.print(f"[bold yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/bold yellow]")
                break

            self.state.current_cycle = self.state.completed_cycles + 1
            logger.info("cycle_start", cycle=self.state.current_cycle)
            console.print(f"\n[bold cyan]═══ Cycle {self.state.current_cycle} ═══[/bold cyan]")
            await self._cycle()
            self.state.completed_cycles += 1
            self._save_state()

            if self._is_done():
                logger.info("goal_complete", total_cycles=self.state.completed_cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Get pending tasks from previous cycles or plan new ones
        pending_tasks = [t for t in self.state.tasks if t.status == "pending"]
        
        if not pending_tasks:
            logger.info("planning_new_tasks")
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal, self.state.tasks)
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
            
            if not plan.tasks:
                logger.info("no_tasks_planned")
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return
            
            # Add new tasks to state
            for task in plan.tasks:
                task.cycle = self.state.current_cycle
                self.state.tasks.append(task)
            pending_tasks = plan.tasks
            logger.info("planned_tasks", count=len(plan.tasks))
        else:
            logger.info("continuing_pending_tasks", count=len(pending_tasks))
            console.print(f"[cyan]Continuing with {len(pending_tasks)} pending tasks from previous cycles[/cyan]")

        # Execute pending tasks in parallel (respect max_parallel_tasks)
        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
        
        async def run_with_semaphore(task: TaskModel):
            async with semaphore:
                logger.info("task_start", task_id=task.id, description=task.description)
                result = await WorkerAgent(task=task, client=self.client).run()
                logger.info("task_complete", task_id=task.id, status="completed")
                return result

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks_to_run = [run_with_semaphore(task) for task in pending_tasks]
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
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

        logger.info("testing")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.state.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Add fix tasks for failures
            self._add_fix_tasks(test_result.failures)

    def _add_fix_tasks(self, failures: list[str]) -> None:
        for i, failure in enumerate(failures):
            task = TaskModel(
                id=f"fix-{self.state.current_cycle}-{i}",
                description=f"Fix test failure: {failure}",
                files=[],
                dependencies=[],
                status="pending",
                cycle=self.state.current_cycle + 1,
            )
            self.state.tasks.append(task)
            logger.info("added_fix_task", task_id=task.id, failure=failure)

    def _is_done(self) -> bool:
        pending = [t for t in self.state.tasks if t.status == "pending"]
        failed = [t for t in self.state.tasks if t.status == "failed"]
        if failed:
            return False
        if not pending:
            return True
        return False

    def _save_state(self) -> None:
        self.state.updated_at = datetime.now().isoformat()
        if not self.state.created_at:
            self.state.created_at = self.state.updated_at
        try:
            settings.state_file.write_text(self.state.model_dump_json(indent=2))
            logger.debug("state_saved", path=str(settings.state_file))
        except Exception as e:
            logger.error("state_save_failed", error=str(e))
            console.print(f"[yellow]Warning: Failed to save state: {e}[/yellow]")

    def _load_state(self) -> None:
        try:
            if settings.state_file.exists():
                data = json.loads(settings.state_file.read_text())
                self.state = State(**data)
                logger.info("state_loaded", cycle=self.state.completed_cycles, tasks=len(self.state.tasks))
                console.print(f"[green]Resumed from cycle {self.state.completed_cycles}[/green]")
        except Exception as e:
            logger.error("state_load_failed", error=str(e))
            console.print(f"[yellow]Warning: Failed to load state: {e}[/yellow]")
