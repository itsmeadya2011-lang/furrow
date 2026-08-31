from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient

console = Console()

MAX_PLAN_RETRIES = 3
MAX_TASK_RETRIES = 2


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.on_output = on_output
        self._failed_cycles = 0

    async def _emit(self, message: str) -> None:
        """Emit output to callback if registered."""
        if self.on_output:
            try:
                await self.on_output(message)
            except Exception:
                pass  # Don't let callback errors break the orchestrator

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit(f"Furrow started\nGoal: {self.goal}\n")

        while True:
            self.cycles += 1
            cycle_header = f"\n═══ Cycle {self.cycles} ═══"
            console.print(f"\n[bold cyan]{cycle_header}[/bold cyan]")
            await self._emit(cycle_header)

            # Check max_cycles before executing
            if settings.max_cycles > 0 and self.cycles > settings.max_cycles:
                msg = f"Max cycles ({settings.max_cycles}) reached. Stopping."
                console.print(f"[yellow]{msg}[/yellow]")
                await self._emit(msg)
                break

            try:
                await self._cycle()
                self._failed_cycles = 0  # Reset on success
            except Exception as e:
                self._failed_cycles += 1
                error_msg = f"Cycle {self.cycles} error: {e}"
                console.print(f"[red]{error_msg}[/red]")
                await self._emit(error_msg)

                if self._failed_cycles >= 3:
                    fatal_msg = "Too many consecutive failures. Stopping."
                    console.print(f"[red]{fatal_msg}[/red]")
                    await self._emit(fatal_msg)
                    break

            if self._is_done():
                msg = "Goal complete. Halting."
                console.print(f"[bold green]{msg}[/bold green]")
                await self._emit(msg)
                break

    async def _cycle(self) -> None:
        # Plan with retry
        plan = await self._plan_with_retry()
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale}\nTasks: {len(plan.tasks)}")

        if not plan.tasks:
            msg = "No tasks planned. Goal may be complete."
            console.print(f"[yellow]{msg}[/yellow]")
            await self._emit(msg)
            return

        # Execute tasks respecting dependencies
        results = await self._execute_tasks_with_dependencies(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(f"Task {task.id} FAILED: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"Task {task.id} completed")

        # Test with error handling
        try:
            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

            if test_result.passed:
                console.print(f"[green]Tests passed: {test_result.summary}[/green]")
                await self._emit(f"Tests PASSED: {test_result.summary}")
            else:
                console.print(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    console.print(f"  • {failure}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
                await self._emit(f"Tests FAILED: {test_result.summary}")
                self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
        except Exception as e:
            console.print(f"[red]Testing error: {e}[/red]")
            await self._emit(f"Testing error: {e}")

    async def _plan_with_retry(self) -> Plan:
        """Plan with retry logic for LLM failures."""
        last_error: Exception | None = None
        for attempt in range(MAX_PLAN_RETRIES):
            try:
                with Status(f"[bold yellow]Planning (attempt {attempt + 1})...", console=console):
                    return await self.planner.plan(self.goal)
            except Exception as e:
                last_error = e
                console.print(f"[yellow]Plan attempt {attempt + 1} failed: {e}[/yellow]")
                if attempt < MAX_PLAN_RETRIES - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        raise RuntimeError(f"Planning failed after {MAX_PLAN_RETRIES} attempts: {last_error}")

    async def _execute_tasks_with_dependencies(
        self, tasks: list[Any]
    ) -> list[Any]:
        """Execute tasks respecting their dependencies.

        Tasks with no dependencies run first, then tasks whose
        dependencies have all completed successfully.
        """
        if not tasks:
            return []

        # If no tasks have dependencies, run all in parallel
        if not any(t.dependencies for t in tasks):
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                coros = [
                    self._run_task_with_retry(task) for task in tasks
                ]
                return await asyncio.gather(*coros, return_exceptions=True)

        # Execute in dependency order
        results: dict[str, Any] = {}
        task_map = {t.id: t for t in tasks}
        remaining = set(t.id for t in tasks)

        with Status("[bold yellow]Executing tasks with dependencies...", console=console):
            while remaining:
                # Find tasks whose dependencies are all completed
                ready = []
                for task_id in remaining:
                    task = task_map[task_id]
                    deps_completed = all(
                        dep in results and not isinstance(results[dep], Exception)
                        for dep in task.dependencies
                    )
                    if deps_completed:
                        ready.append(task)

                if not ready:
                    # Circular dependency or failed dependency - mark remaining as failed
                    for task_id in remaining:
                        results[task_id] = Exception(
                            "Unresolved dependencies (circular or failed)"
                        )
                    break

                # Execute ready tasks in parallel
                coros = [self._run_task_with_retry(task) for task in ready]
                batch_results = await asyncio.gather(*coros, return_exceptions=True)

                for task, result in zip(ready, batch_results):
                    results[task.id] = result
                    remaining.discard(task.id)

        return [results[t.id] for t in tasks]

    async def _run_task_with_retry(self, task: Any) -> Any:
        """Run a single task with retry logic."""
        last_error: Exception | None = None
        for attempt in range(MAX_TASK_RETRIES):
            try:
                return await WorkerAgent(task=task, client=self.client).run()
            except Exception as e:
                last_error = e
                console.print(
                    f"[yellow]Task {task.id} attempt {attempt + 1} failed: {e}[/yellow]"
                )
                if attempt < MAX_TASK_RETRIES - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        return last_error

    def _is_done(self) -> bool:
        """Check if all tasks in the current plan are completed."""
        if self.current_plan is None:
            return False
        tasks = self.current_plan.tasks
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        """Get tasks from current plan for backward compatibility."""
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
