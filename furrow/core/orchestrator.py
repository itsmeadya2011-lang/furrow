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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

configure_logging()
console = Console()
logger = get_logger(__name__)

STATE_FILE = ".furrow_state.json"
FEEDBACK_FILE = ".furrow_feedback.txt"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Any | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.settings = settings or self.client.settings
        self._task_id_counter = 0
        self._load_state()

    def _load_state(self) -> None:
        state_path = self.settings.workspace / STATE_FILE
        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = json.load(f)
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self._task_id_counter = data.get("task_id_counter", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                logger.info("state_loaded", tasks=len(self.all_tasks), cycle=self.cycles)
                console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/dim]")
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        state_path = self.settings.workspace / STATE_FILE
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "task_id_counter": self._task_id_counter,
            "tasks": [t.model_dump() for t in self.all_tasks],
        }
        try:
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("state_saved", tasks=len(self.all_tasks))
        except Exception as e:
            logger.warning("state_save_failed", error=str(e))
            console.print(f"[yellow]Could not save state: {e}[/yellow]")

    def _read_feedback(self) -> str | None:
        """Read user feedback from feedback file."""
        feedback_path = self.settings.workspace / FEEDBACK_FILE
        if feedback_path.exists():
            try:
                with open(feedback_path) as f:
                    feedback = f.read().strip()
                if feedback:
                    # Clear feedback file after reading
                    feedback_path.unlink()
                    return feedback
            except Exception as e:
                logger.warning("feedback_read_failed", error=str(e))
        return None

    def _incorporate_feedback(self, feedback: str) -> None:
        """Incorporate user feedback into the current goal."""
        self.goal = f"{self.goal}\n\nUser feedback: {feedback}"
        logger.info("feedback_incorporated", feedback=feedback[:100])

    async def run(self) -> None:
        logger.info("orchestrator_started", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.settings.max_cycles and self.cycles > self.settings.max_cycles:
                logger.info("max_cycles_reached", max_cycles=self.settings.max_cycles)
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                logger.info("goal_completed", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                # Clean up state file on completion
                state_path = self.settings.workspace / STATE_FILE
                if state_path.exists():
                    state_path.unlink()
                break

    async def _cycle(self) -> None:
        # Check for user feedback at the start of each cycle
        feedback = self._read_feedback()
        if feedback:
            self._incorporate_feedback(feedback)
            console.print(f"[cyan]Incorporated user feedback: {feedback[:100]}...[/cyan]")

        logger.info("planning_started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("planning_completed", tasks=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Assign unique IDs to new tasks and add to tracking
        for task in plan.tasks:
            self._task_id_counter += 1
            if not task.id or task.id in [str(t.id) for t in self.all_tasks]:
                task.id = str(self._task_id_counter)
            self.all_tasks.append(task)

        # Execute tasks with retry logic
        await self._execute_tasks_with_retry(plan.tasks)

        logger.info("testing_started")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks_with_retry(self, tasks: list[TaskModel]) -> None:
        """Execute tasks with retry logic for failed tasks."""
        pending_tasks = list(tasks)
        
        while pending_tasks:
            logger.info("execution_batch_started", tasks=len(pending_tasks))
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                worker_tasks = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in pending_tasks
                ]
                results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Process results and identify tasks that need retry
            still_pending = []
            for task, result in zip(pending_tasks, results):
                if isinstance(result, Exception):
                    task.retry_count += 1
                    task.status = "failed"
                    task.result = str(result)
                    logger.error("task_failed", task_id=task.id, error=str(result), retry_count=task.retry_count)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    
                    if task.retry_count < self.settings.max_task_retries:
                        logger.info("task_scheduled_for_retry", task_id=task.id, retry_count=task.retry_count)
                        console.print(f"[yellow]Task {task.id} will be retried (attempt {task.retry_count + 1}/{self.settings.max_task_retries})[/yellow]")
                        task.status = "pending"  # Reset for retry
                        still_pending.append(task)
                    else:
                        logger.error("task_max_retries_exceeded", task_id=task.id)
                        console.print(f"[red]Task {task.id} exceeded max retries ({self.settings.max_task_retries})[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    logger.info("task_completed", task_id=task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
            
            pending_tasks = still_pending

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed" and t.retry_count < self.settings.max_task_retries)
        permanently_failed = sum(1 for t in self.all_tasks if t.status == "failed" and t.retry_count >= self.settings.max_task_retries)
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        # Tasks that have exceeded max retries don't block completion
        active_tasks = len(self.all_tasks) - permanently_failed
        
        if failed > 0:
            return False
        if pending > 0:
            return False
        if completed >= active_tasks:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
