from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.progress_callback = progress_callback

    def _emit(self, message: str) -> None:
        console.print(message)
        if self.progress_callback:
            self.progress_callback(message)

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        while True:
            # Check max cycles
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._emit(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break
            
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            cycle_complete = await self._cycle()
            if cycle_complete:
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> bool:
        # Plan
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return True

        # Limit parallel tasks
        tasks_to_run = plan.tasks[: self.settings.max_parallel_tasks]
        if len(plan.tasks) > self.settings.max_parallel_tasks:
            self._emit(f"[yellow]Limiting to {self.settings.max_parallel_tasks} parallel tasks (planned {len(plan.tasks)})[/yellow]")

        # Execute in parallel
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Process results
        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")
            
            # Add to all tasks for tracking
            self.all_tasks.append(task)

        # Test
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, tasks_to_run)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to focus on fixing failures
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        return self._is_done()

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        # If there are failed tasks, not done (will retry)
        if failed > 0:
            return False
        
        # If all tasks completed and no pending, done
        if completed > 0 and pending == 0 and failed == 0:
            return True
            
        return False

    def get_state(self) -> dict[str, Any]:
        """Get current state for persistence."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.all_tasks],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], client: LLMClient | None = None, settings: Settings | None = None) -> "Orchestrator":
        """Restore orchestrator from saved state."""
        orchestrator = cls(goal=state["goal"], client=client, settings=settings)
        orchestrator.cycles = state["cycles"]
        orchestrator.all_tasks = [TaskModel(**t) for t in state.get("tasks", [])]
        return orchestrator
