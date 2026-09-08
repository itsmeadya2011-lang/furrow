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
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            # Check max cycles
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                break
            
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Build context from previous cycles
        context = self._build_context()
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context=context)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track new tasks
        for task in plan.tasks:
            task.status = "pending"
            self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            # Limit parallel tasks
            semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
            
            async def run_with_semaphore(task: TaskModel) -> str:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()
            
            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in plan.tasks],
            "test_result": test_result.model_dump(),
        })

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal with test failures for next cycle
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            
        self._print_progress()

    def _build_context(self) -> str:
        """Build context string from previous cycles for the planner."""
        if not self.cycle_history:
            return ""
        
        parts = ["## Previous Cycles Summary"]
        for cycle in self.cycle_history[-3:]:  # Last 3 cycles
            parts.append(f"\n### Cycle {cycle['cycle']}")
            parts.append(f"Goal: {cycle['goal'][:200]}...")
            for task in cycle['tasks']:
                parts.append(f"  - Task {task['id']}: {task['description'][:100]} ({task['status']})")
            if cycle['test_result']['passed']:
                parts.append(f"  Tests: PASSED - {cycle['test_result']['summary']}")
            else:
                parts.append(f"  Tests: FAILED - {cycle['test_result']['summary']}")
                for failure in cycle['test_result']['failures']:
                    parts.append(f"    - {failure}")
        return "\n".join(parts)

    def _print_progress(self) -> None:
        """Print a summary table of all tasks."""
        table = Table(title="Task Progress")
        table.add_column("Cycle", style="cyan")
        table.add_column("Task ID", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        
        for task in self.all_tasks:
            status_style = "green" if task.status == "completed" else "red" if task.status == "failed" else "yellow"
            table.add_row(
                str(self.cycles),
                task.id,
                task.description[:60],
                f"[{status_style}]{task.status}[/{status_style}]"
            )
        console.print(table)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        
        # If any task failed, not done (will retry)
        if failed > 0:
            return False
        
        # All tasks completed
        if completed >= len(self.all_tasks):
            return True
        
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
