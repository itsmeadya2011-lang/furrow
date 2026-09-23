from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger(__name__)


def topological_sort(tasks: list[TaskModel]) -> list[list[TaskModel]]:
    """Sort tasks into levels based on dependencies. Returns list of levels (each level can run in parallel)."""
    # Build adjacency list and in-degree count
    task_map = {task.id: task for task in tasks}
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    
    for task in tasks:
        for dep_id in task.dependencies:
            if dep_id in task_map:
                adj[dep_id].append(task.id)
                in_degree[task.id] += 1
            else:
                logger.warning("task.dependency_not_found", task_id=task.id, dependency=dep_id)
    
    # Kahn's algorithm for topological sorting
    queue = deque([task.id for task in tasks if in_degree[task.id] == 0])
    levels = []
    
    while queue:
        level = []
        level_size = len(queue)
        
        for _ in range(level_size):
            task_id = queue.popleft()
            level.append(task_map[task_id])
            
            for neighbor in adj[task_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if level:
            levels.append(level)
    
    # Check for cycles
    total_sorted = sum(len(level) for level in levels)
    if total_sorted < len(tasks):
        logger.warning("task.circular_dependency_detected", remaining=len(tasks) - total_sorted)
        # Add remaining tasks to a final level
        remaining = [task_map[tid] for tid in task_map if tid not in {t.id for level in levels for t in level}]
        if remaining:
            levels.append(remaining)
    
    return levels


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_tasks: set[str] = set()
        self.failed_tasks: dict[str, str] = {}
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        logger.info("furrow.started", goal=self.original_goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        max_cycles = settings.max_cycles if settings.max_cycles > 0 else float("inf")

        while self.cycles < max_cycles:
            self.cycles += 1
            logger.info("cycle.started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            cycle_result = await self._cycle()
            self.cycle_history.append(cycle_result)
            
            if self._is_done():
                logger.info("furrow.completed", cycles=self.cycles, total_tasks=len(self.all_tasks))
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            
            if self.cycles >= max_cycles:
                logger.warning("furrow.max_cycles_reached", max_cycles=max_cycles)
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Stopping.[/yellow]")
                break

        self._print_summary()

    async def _cycle(self) -> dict[str, Any]:
        cycle_data = {"cycle": self.cycles, "tasks_planned": 0, "tasks_completed": 0, "tasks_failed": 0, "tests_passed": False}

        # Plan with context of previous cycles
        logger.info("planning.started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self._get_context())
        logger.info("planning.completed", cycle=self.cycles, tasks_planned=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("planning.no_tasks", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return cycle_data

        # Filter out already completed tasks
        new_tasks = [t for t in plan.tasks if t.id not in self.completed_tasks]
        if not new_tasks:
            logger.info("planning.all_completed", cycle=self.cycles)
            console.print("[green]All planned tasks already completed.[/green]")
            return cycle_data

        cycle_data["tasks_planned"] = len(new_tasks)

        # Add new tasks to our tracking
        for task in new_tasks:
            if task.id not in [t.id for t in self.all_tasks]:
                self.all_tasks.append(task)

        # Execute tasks in parallel with dependency resolution
        logger.info("execution.started", cycle=self.cycles, tasks=len(new_tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await self._execute_tasks(new_tasks)

        for task, result in zip(new_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_tasks[task.id] = str(result)
                cycle_data["tasks_failed"] += 1
                logger.error("task.failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_tasks.add(task.id)
                cycle_data["tasks_completed"] += 1
                logger.info("task.completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        # Test
        logger.info("testing.started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.original_goal, self.all_tasks)

        cycle_data["tests_passed"] = test_result.passed

        if test_result.passed:
            logger.info("testing.passed", cycle=self.cycles, summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("testing.failed", cycle=self.cycles, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to focus on fixing failures while preserving original goal
            self.goal = f"{self.original_goal}\n\nPrevious test failures to fix:\n" + "\n".join(test_result.failures)

        return cycle_data

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        # Respect max_parallel_tasks setting
        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

        async def run_with_retry(task: TaskModel) -> Any:
            async with semaphore:
                for attempt in range(settings.max_retries + 1):
                    if attempt > 0:
                        logger.info("task.retry", task_id=task.id, attempt=attempt)
                        await asyncio.sleep(settings.retry_delay * attempt)
                    
                    worker = WorkerAgent(task=task, client=self.client)
                    result = await worker.run()
                    
                    if not isinstance(result, Exception):
                        return result
                    
                    logger.warning("task.attempt_failed", task_id=task.id, attempt=attempt + 1, error=str(result))
                
                return Exception(f"Task {task.id} failed after {settings.max_retries + 1} attempts")

        # Sort tasks by dependencies
        levels = topological_sort(tasks)
        logger.info("execution.dependency_levels", levels=len(levels), total_tasks=len(tasks))
        
        all_results = {}
        
        for level_idx, level in enumerate(levels):
            logger.info("execution.level_started", level=level_idx + 1, tasks=len(level))
            with Status(f"[bold yellow]Executing level {level_idx + 1}/{len(levels)}...", console=console):
                level_results = await asyncio.gather(
                    *[run_with_retry(task) for task in level],
                    return_exceptions=True
                )
            
            for task, result in zip(level, level_results):
                all_results[task.id] = result

        # Return results in original task order
        return [all_results.get(task.id, Exception(f"Task {task.id} not executed")) for task in tasks]

    def _get_context(self) -> str:
        """Build context from previous cycles for the planner."""
        if not self.cycle_history:
            return "First cycle - no previous work."

        context_parts = ["Previous cycles summary:"]
        for cycle in self.cycle_history:
            context_parts.append(
                f"Cycle {cycle['cycle']}: Planned {cycle['tasks_planned']} tasks, "
                f"Completed {cycle['tasks_completed']}, Failed {cycle['tasks_failed']}, "
                f"Tests: {'PASSED' if cycle['tests_passed'] else 'FAILED'}"
            )

        # Add completed task details
        completed = [t for t in self.all_tasks if t.status == "completed"]
        if completed:
            context_parts.append("\nCompleted tasks:")
            for task in completed:
                context_parts.append(f"  - {task.id}: {task.description} -> {task.result[:100] if task.result else 'done'}")

        # Add failed task details
        if self.failed_tasks:
            context_parts.append("\nFailed tasks:")
            for task_id, error in self.failed_tasks.items():
                context_parts.append(f"  - {task_id}: {error[:200]}")

        return "\n".join(context_parts)

    def _is_done(self) -> bool:
        # All tasks completed and tests passed
        pending = [t for t in self.all_tasks if t.status == "pending"]
        failed = [t for t in self.all_tasks if t.status == "failed"]
        
        if failed:
            return False
        if not pending and self.all_tasks:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def _print_summary(self) -> None:
        table = Table(title="Furrow Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Cycles", str(self.cycles))
        table.add_row("Total Tasks", str(len(self.all_tasks)))
        table.add_row("Completed", str(len(self.completed_tasks)))
        table.add_row("Failed", str(len(self.failed_tasks)))
        table.add_row("Pending", str(len([t for t in self.all_tasks if t.status == "pending"])))

        console.print(table)