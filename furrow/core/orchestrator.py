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
from furrow.config import Plan, TestResult, Settings, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = self.settings.workspace / ".furrow_state.json"
        self.max_retries = 3
        self._load_state()

    def _load_state(self) -> None:
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                self.goal = state.get("goal", self.goal)
                self.cycles = state.get("cycles", 0)
                # Restore tasks from state
                for task_data in state.get("tasks", []):
                    task = TaskModel(**task_data)
                    self.all_tasks.append(task)
                console.print(f"[green]Loaded state from {self.state_file}[/green]")
                logger.info("state_loaded", state_file=str(self.state_file), cycles=self.cycles, task_count=len(self.all_tasks))
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        """Save current state to file."""
        try:
            state = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [task.model_dump() for task in self.all_tasks],
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug("state_saved", state_file=str(self.state_file), cycles=self.cycles, task_count=len(self.all_tasks))
        except Exception as e:
            console.print(f"[yellow]Could not save state: {e}[/yellow]")
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal, max_cycles=self.settings.max_cycles)
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=self.settings.max_cycles)
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_completed", cycles=self.cycles)
                # Clean up state file on completion
                if self.state_file.exists():
                    self.state_file.unlink()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        # Track all tasks across cycles
        for task in plan.tasks:
            self.all_tasks.append(task)

        # Execute tasks with retry logic
        await self._execute_tasks_with_retry(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks_with_retry(self, tasks: list[TaskModel]) -> None:
        """Execute tasks with retry logic for failures."""
        pending_tasks = list(tasks)

        for attempt in range(self.max_retries):
            if not pending_tasks:
                break

            if attempt > 0:
                console.print(f"[yellow]Retry attempt {attempt}/{self.max_retries - 1} for {len(pending_tasks)} failed task(s)[/yellow]")
                logger.info("retry_attempt", attempt=attempt, pending_count=len(pending_tasks))

            with Status(f"[bold yellow]Executing tasks (attempt {attempt + 1})...", console=console):
                worker_tasks = [
                    WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    for task in pending_tasks
                ]
                results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            pending_tasks = []
            for task, result in zip(pending_tasks if attempt > 0 else tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("task_failed", task_id=task.id, error=str(result), attempt=attempt)
                    if attempt < self.max_retries - 1:
                        pending_tasks.append(task)
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False
