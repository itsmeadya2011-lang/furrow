from __future__ import annotations

import asyncio
import json
import os
import signal
import time
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
from furrow.config import Plan, Settings, State, TaskModel, TestResult, settings
from furrow.llm import LLMClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(structlog, settings.log_level.upper())
    ),
)
logger = structlog.get_logger()

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings = settings,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.planner = PlannerAgent(client=self.client, settings=settings)
        self.tester = TesterAgent(client=self.client, settings=settings)
        self.cycles = 0
        self._shutdown = False
        self._state_path = state_path or (settings.workspace / settings.state_file)
        self._state = State(goal=goal)

        # Setup signal handlers for graceful shutdown
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

    def _request_shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info("shutdown_requested")
        console.print("\n[yellow]Shutdown requested. Finishing current cycle...[/yellow]")
        self._shutdown = True

    async def run(self) -> None:
        # Load existing state if available
        loaded_state = State.load(self._state_path)
        if loaded_state and loaded_state.goal == self.goal:
            self._state = loaded_state
            self.cycles = self._state.cycles
            console.print(f"[cyan]Resumed from cycle {self.cycles}[/cyan]")
        else:
            self._state = State(goal=self.goal)

        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while not self._shutdown:
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            self._state.cycles = self.cycles
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            # Save state after each cycle
            self._state.updated_at = time.time()
            self._state.save(self._state_path)

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.info("cycle_start", cycle=self.cycles)

        # Get pending tasks that are ready to run (dependencies satisfied)
        ready_tasks = self._get_ready_tasks()
        if not ready_tasks:
            # Plan new tasks if none ready
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal, self._get_context())
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return

            # Add new tasks to state
            for task in plan.tasks:
                task.status = "pending"
                self._state.tasks.append(task)

            # Re-evaluate ready tasks
            ready_tasks = self._get_ready_tasks()

        if not ready_tasks:
            console.print("[yellow]No tasks ready to execute. Waiting for dependencies...[/yellow]")
            return

        # Execute ready tasks in parallel (respecting max_parallel_tasks)
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                return task, await worker.run()

        with Status(f"[bold yellow]Executing {len(ready_tasks)} tasks in parallel...", console=console):
            tasks_coro = [run_task_with_semaphore(task) for task in ready_tasks]
            results = await asyncio.gather(*tasks_coro, return_exceptions=True)

        # Process results
        for task, result in zip(ready_tasks, results):
            if isinstance(result, Exception):
                await self._handle_task_failure(task, result)
            else:
                self._handle_task_success(task, result)

        # Run tests
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, self._state.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary, cycle=self.cycles)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures, cycle=self.cycles)
            # Update goal to focus on fixing tests
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            self._state.goal = self.goal

        logger.info("cycle_complete", cycle=self.cycles, completed=len(self._state.completed_tasks), failed=len(self._state.failed_tasks))

    def _get_ready_tasks(self) -> list[TaskModel]:
        """Get tasks that are pending and have all dependencies satisfied."""
        completed_ids = {t.id for t in self._state.completed_tasks}
        ready = []
        for task in self._state.tasks:
            if task.status != "pending":
                continue
            if all(dep in completed_ids for dep in task.dependencies):
                ready.append(task)
        return ready

    def _get_context(self) -> str:
        """Build context string for planner from completed work."""
        if not self._state.completed_tasks:
            return ""
        parts = ["Completed work:"]
        for task in self._state.completed_tasks:
            parts.append(f"- {task.id}: {task.description} → {task.result[:200]}")
        return "\n".join(parts)

    def _handle_task_success(self, task: TaskModel, result: str) -> None:
        task.status = "completed"
        task.result = result
        task.completed_at = time.time()
        self._state.completed_tasks.append(task)
        self._state.tasks.remove(task)
        console.print(f"[green]Task {task.id} completed[/green]")
        logger.info("task_completed", task_id=task.id, cycle=self.cycles)

    async def _handle_task_failure(self, task: TaskModel, error: Exception) -> None:
        task.status = "failed"
        task.result = str(error)
        task.retry_count += 1

        max_retries = 3
        if task.retry_count < max_retries:
            # Retry with exponential backoff
            backoff = 2 ** task.retry_count
            console.print(f"[yellow]Task {task.id} failed (attempt {task.retry_count}/{max_retries}). Retrying in {backoff}s...[/yellow]")
            logger.warning("task_retry", task_id=task.id, attempt=task.retry_count, error=str(error))
            await asyncio.sleep(backoff)
            task.status = "pending"  # Reset for retry
        else:
            self._state.failed_tasks.append(task)
            self._state.tasks.remove(task)
            console.print(f"[red]Task {task.id} failed permanently after {max_retries} retries: {error}[/red]")
            logger.error("task_failed_permanently", task_id=task.id, error=str(error))

    def _is_done(self) -> bool:
        # Check if all tasks are completed and no pending tasks remain
        has_pending = any(t.status == "pending" for t in self._state.tasks)
        has_failed = len(self._state.failed_tasks) > 0
        has_running = any(t.status == "running" for t in self._state.tasks)

        if has_failed and not has_pending and not has_running:
            # All tasks done but some failed permanently
            return True

        if not has_pending and not has_running and self._state.completed_tasks:
            # All planned work done
            return True

        return False

    def _get_tasks(self) -> list[Any]:
        return self._state.tasks
