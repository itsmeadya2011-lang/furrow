from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

console = Console()
logger = get_logger("orchestrator")

# Configure logging
configure_logging(settings.log_level)

STATE_FILE = ".furrow_state.json"
MAX_RETRIES = 3


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []  # Persistent task list across cycles
        self.start_time = time.time()
        self.state_file = Path.cwd() / STATE_FILE
        self.task_retries: dict[str, int] = {}  # Track retries per task
        
        if resume:
            self._load_state()

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                self.cycles += 1
                logger.info("cycle_start", cycle=self.cycles)
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                self._save_state()
                if self._is_done():
                    logger.info("goal_complete", cycles=self.cycles)
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
                if self._should_stop():
                    logger.warning("max_cycles_or_timeout", cycles=self.cycles)
                    console.print("[yellow]Max cycles reached or timeout. Stopping.[/yellow]")
                    break
        except KeyboardInterrupt:
            logger.info("interrupted", cycles=self.cycles)
            console.print("\n[yellow]Interrupted. Saving state...[/yellow]")
            self._save_state()
            raise

    async def _cycle(self) -> None:
        logger.info("planning_start", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            # Include context from previous tasks
            context = self._build_context()
            plan = await self.planner.plan(self.goal, context)
        logger.info("planning_complete", cycle=self.cycles, task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to persistent list
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        logger.info("execution_start", cycle=self.cycles)
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            # Run pending tasks and retry failed ones
            pending_tasks = [t for t in plan.tasks if t.status == "pending"]
            failed_tasks = [t for t in plan.tasks if t.status == "failed" and self.task_retries.get(t.id, 0) < MAX_RETRIES]
            
            # Combine pending and retryable failed tasks
            tasks_to_run = pending_tasks + failed_tasks
            
            if not tasks_to_run:
                logger.info("no_tasks_to_run")
                console.print("[yellow]All tasks completed or max retries reached.[/yellow]")
                return
            
            # Run tasks with retry logic
            for task in tasks_to_run:
                await self._run_task_with_retry(task)

        logger.info("testing_start", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_task_with_retry(self, task: TaskModel) -> None:
        """Run a single task with retry logic."""
        max_retries = MAX_RETRIES
        current_retry = self.task_retries.get(task.id, 0)
        
        logger.info("task_start", task_id=task.id, description=task.description, retry=current_retry)
        
        while current_retry <= max_retries:
            if current_retry > 0:
                logger.info("task_retry", task_id=task.id, attempt=current_retry, max_retries=max_retries)
                console.print(f"[yellow]Retrying task {task.id} (attempt {current_retry}/{max_retries})[/yellow]")
            
            worker = WorkerAgent(task=task, client=self.client)
            result = await worker.run()
            
            # Check if task succeeded (worker returns string, not exception)
            if "error" not in result.lower() and "failed" not in result.lower():
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                self.task_retries.pop(task.id, None)
                return
            else:
                current_retry += 1
                self.task_retries[task.id] = current_retry
                task.status = "failed"
                task.result = result
                logger.warning("task_failed", task_id=task.id, attempt=current_retry, result=result[:200])
                console.print(f"[red]Task {task.id} failed (attempt {current_retry}): {result[:200]}[/red]")
                
                if current_retry > max_retries:
                    logger.error("task_max_retries_exceeded", task_id=task.id)
                    console.print(f"[red]Task {task.id} exceeded max retries[/red]")
                    return

    def _build_context(self) -> str:
        """Build context from previous tasks for the planner."""
        if not self.all_tasks:
            return "No previous tasks."
        
        lines = ["Previous tasks:"]
        for task in self.all_tasks:
            lines.append(f"  Task {task.id}: {task.description} - {task.status}")
            if task.result:
                lines.append(f"    Result: {task.result[:200]}")
        return "\n".join(lines)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        if failed > 0:
            return False
        if pending == 0 and completed > 0:
            return True
        return False

    def _should_stop(self) -> bool:
        """Check if we should stop due to max cycles or timeout."""
        if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
            return True
        # Optional: add timeout check (e.g., 1 hour)
        if time.time() - self.start_time > 3600:
            return True
        return False

    def get_task_summary(self) -> dict:
        """Get a summary of all tasks for reporting."""
        return {
            "total": len(self.all_tasks),
            "completed": sum(1 for t in self.all_tasks if t.status == "completed"),
            "failed": sum(1 for t in self.all_tasks if t.status == "failed"),
            "pending": sum(1 for t in self.all_tasks if t.status == "pending"),
            "cycles": self.cycles,
        }

    def _save_state(self) -> None:
        """Save orchestrator state to disk."""
        try:
            state = {
                "goal": self.goal,
                "cycles": self.cycles,
                "start_time": self.start_time,
                "tasks": [task.model_dump() for task in self.all_tasks],
                "task_retries": self.task_retries,
            }
            self.state_file.write_text(json.dumps(state, indent=2))
            logger.debug("state_saved", cycles=self.cycles, task_count=len(self.all_tasks))
        except Exception as e:
            logger.error("state_save_failed", error=str(e))
            console.print(f"[red]Failed to save state: {e}[/red]")

    def _load_state(self) -> None:
        """Load orchestrator state from disk."""
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.start_time = data.get("start_time", time.time())
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.task_retries = data.get("task_retries", {})
                logger.info("state_loaded", cycles=self.cycles, task_count=len(self.all_tasks))
                console.print(f"[green]Resumed from cycle {self.cycles} with {len(self.all_tasks)} tasks[/green]")
        except Exception as e:
            logger.warning("state_load_failed", error=str(e))
            console.print(f"[yellow]Could not load state: {e}[/yellow]")
