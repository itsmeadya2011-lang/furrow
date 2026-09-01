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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.core.state import StateManager
from furrow.llm import LLMClient

console = Console()


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class PlanningError(OrchestratorError):
    """Raised when planning fails."""
    pass


class ExecutionError(OrchestratorError):
    """Raised when task execution fails."""
    pass


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_manager: StateManager | None = None,
        max_retries: int = 3,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._completed_goals: list[str] = []
        self._state_manager = state_manager
        self._session_id: str | None = None
        self._max_retries = max_retries
        self._failed_tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        # Start session tracking if state manager is available
        if self._state_manager:
            self._session_id = self._state_manager.start_session(self.goal)
            console.print(f"[dim]Session ID: {self._session_id}[/dim]")

        try:
            while True:
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()

                # Update session state
                if self._state_manager and self._session_id:
                    self._state_manager.update_session(
                        self._session_id,
                        cycles=self.cycles,
                    )

                # Check max_cycles limit (0 = infinite)
                if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                    console.print(f"[yellow]Reached max cycles ({settings.max_cycles}). Stopping.[/yellow]")
                    if self._state_manager and self._session_id:
                        self._state_manager.complete_session(self._session_id, "max_cycles_reached")
                    break

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    self._completed_goals.append(self.goal)
                    if self._state_manager and self._session_id:
                        self._state_manager.complete_session(self._session_id, "completed")
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            if self._state_manager and self._session_id:
                self._state_manager.complete_session(self._session_id, "interrupted")
        except Exception as e:
            console.print(f"\n[red]Fatal error: {e}[/red]")
            if self._state_manager and self._session_id:
                self._state_manager.complete_session(self._session_id, f"error: {e}")
            raise

    async def _cycle(self) -> None:
        # Planning with retry
        plan = await self._plan_with_retry()
        self._current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Limit parallel tasks to max_parallel_tasks setting
        tasks_to_run = plan.tasks[: settings.max_parallel_tasks]
        if len(plan.tasks) > settings.max_parallel_tasks:
            console.print(f"[yellow]Limiting to {settings.max_parallel_tasks} parallel tasks[/yellow]")

        # Execute tasks with error handling
        await self._execute_tasks(tasks_to_run)

        # Testing with error handling
        await self._run_tests(tasks_to_run)

    async def _plan_with_retry(self) -> Plan:
        """Plan with retry logic for transient failures."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                with Status("[bold yellow]Planning...", console=console) as status:
                    return await self.planner.plan(self.goal)
            except Exception as e:
                last_error = e
                console.print(f"[yellow]Planning attempt {attempt} failed: {e}[/yellow]")
                if attempt < self._max_retries:
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    console.print(f"[dim]Retrying in {wait_time}s...[/dim]")
                    await asyncio.sleep(wait_time)

        raise PlanningError(f"Planning failed after {self._max_retries} attempts: {last_error}")

    async def _execute_tasks(self, tasks_to_run: list[TaskModel]) -> None:
        """Execute tasks with comprehensive error handling."""
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            task_coroutines = [
                self._execute_single_task(task) for task in tasks_to_run
            ]
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._failed_tasks.append(task)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                if self._state_manager and self._session_id:
                    self._state_manager.add_task_result(
                        self._session_id, task.id, "failed", task.description
                    )
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                if self._state_manager and self._session_id:
                    self._state_manager.add_task_result(
                        self._session_id, task.id, "completed", task.description
                    )

    async def _execute_single_task(self, task: TaskModel) -> str:
        """Execute a single task with retry logic."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await WorkerAgent(task=task, client=self.client).run()
            except Exception as e:
                last_error = e
                console.print(f"[yellow]Task {task.id} attempt {attempt} failed: {e}[/yellow]")
                if attempt < self._max_retries:
                    await asyncio.sleep(1)

        raise ExecutionError(f"Task {task.id} failed after {self._max_retries} attempts: {last_error}")

    async def _run_tests(self, tasks_to_run: list[TaskModel]) -> None:
        """Run tests with error handling."""
        try:
            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await TesterAgent(client=self.client).run(self.goal, tasks_to_run)

            if test_result.passed:
                console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            else:
                console.print(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    console.print(f"  • {failure}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
                self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
        except Exception as e:
            console.print(f"[red]Test execution error: {e}[/yellow]")
            console.print("[yellow]Will retry tests in next cycle.[/yellow]")

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        if self._current_plan is None:
            return []
        return self._current_plan.tasks

    def get_failed_tasks(self) -> list[TaskModel]:
        """Get list of failed tasks for inspection."""
        return self._failed_tasks.copy()
