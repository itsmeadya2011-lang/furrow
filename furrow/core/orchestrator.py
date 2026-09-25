from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


@dataclass
class OrchestratorState:
    """Tracks state across orchestrator cycles."""
    all_tasks: list[TaskModel] = field(default_factory=list)
    completed_tasks: list[TaskModel] = field(default_factory=list)
    failed_tasks: list[TaskModel] = field(default_factory=list)
    original_goal: str = ""
    current_goal: str = ""
    cycle: int = 0
    total_cost: float = 0.0


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.state = OrchestratorState(original_goal=goal, current_goal=goal)
        self._max_cycles = self.settings.max_cycles

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.state.original_goal}", title="Furrow"))
        
        while not self._is_done():
            self.state.cycle += 1
            
            if self._max_cycles > 0 and self.state.cycle > self._max_cycles:
                console.print(f"[bold yellow]Max cycles ({self._max_cycles}) reached. Stopping.[/bold yellow]")
                break
                
            console.print(f"\n[bold cyan]═══ Cycle {self.state.cycle} ═══[/bold cyan]")
            await self._cycle()
            
        self._print_final_summary()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.state.current_goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to state
        for task in plan.tasks:
            # Check if task already exists
            existing = next((t for t in self.state.all_tasks if t.id == task.id), None)
            if not existing:
                self.state.all_tasks.append(task)
            else:
                # Update existing task with new plan info
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
                existing.status = "pending"
                existing.result = None

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                if task not in self.state.failed_tasks:
                    self.state.failed_tasks.append(task)
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                if task not in self.state.completed_tasks:
                    self.state.completed_tasks.append(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.state.original_goal, self.state.all_tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            # Check if all tasks are complete
            if self._all_tasks_complete():
                console.print("[bold green]All tasks complete and tests passing![/bold green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.state.current_goal = f"Fix failing tests for original goal: {self.state.original_goal}\nFailures:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        """Check if the orchestrator should stop."""
        if self._max_cycles > 0 and self.state.cycle >= self._max_cycles:
            return True
            
        # Check if all tasks are completed and tests would pass
        return self._all_tasks_complete()

    def _all_tasks_complete(self) -> bool:
        """Check if all known tasks are completed (not failed)."""
        if not self.state.all_tasks:
            return False
            
        for task in self.state.all_tasks:
            if task.status != "completed":
                return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        """Get all tasks across all cycles."""
        return self.state.all_tasks

    def _print_final_summary(self) -> None:
        console.print("\n[bold cyan]═══ Final Summary ═══[/bold cyan]")
        console.print(f"Original Goal: {self.state.original_goal}")
        console.print(f"Cycles Run: {self.state.cycle}")
        console.print(f"Total Tasks: {len(self.state.all_tasks)}")
        console.print(f"Completed: {len(self.state.completed_tasks)}")
        console.print(f"Failed: {len(self.state.failed_tasks)}")
        
        if self.state.failed_tasks:
            console.print("\n[red]Failed Tasks:[/red]")
            for task in self.state.failed_tasks:
                console.print(f"  • {task.id}: {task.description}")
                if task.result:
                    console.print(f"    Error: {task.result[:200]}")

    def get_state(self) -> OrchestratorState:
        """Return current state for serialization/web UI."""
        return self.state