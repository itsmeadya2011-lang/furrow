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
from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.llm import LLMClient

console = Console()

# Type for progress callback
ProgressCallback = Callable[[str], Awaitable[None]]


class Orchestrator:
    STATE_FILE = ".furrow_state.json"

    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None, 
        max_cycles: int = 0,
        on_progress: ProgressCallback | None = None,
        workspace: Path | None = None,
        resume: bool = False
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.all_tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.on_progress = on_progress
        self.workspace = workspace or self.client.settings.workspace
        self.state_file = self.workspace / self.STATE_FILE
        
        if resume:
            self._load_state()

    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.completed_task_ids = set(data.get("completed_task_ids", []))
                
                # Restore tasks
                for task_data in data.get("tasks", []):
                    task = TaskModel(**task_data)
                    self.all_tasks.append(task)
                
                console.print(f"[green]Resumed from state: cycle {self.cycles}, {len(self.completed_task_ids)} tasks completed[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        """Save state to file."""
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "completed_task_ids": list(self.completed_task_ids),
                "tasks": [t.model_dump() for t in self.all_tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            console.print(f"[yellow]Could not save state: {e}[/yellow]")

    async def _emit(self, message: str) -> None:
        """Emit progress message to callback and console."""
        console.print(message)
        if self.on_progress:
            await self.on_progress(message)

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles and self.cycles >= self.max_cycles:
                await self._emit(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                # Clean up state file on completion
                if self.state_file.exists():
                    self.state_file.unlink()
                break

    async def _cycle(self) -> None:
        await self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal, self.all_tasks)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed tasks
        new_tasks = [t for t in plan.tasks if t.id not in self.completed_task_ids]
        if not new_tasks:
            await self._emit("[yellow]All planned tasks already completed.[/yellow]")
            return

        # Add new tasks to our tracking
        for task in new_tasks:
            self.all_tasks.append(task)

        # Execute tasks respecting dependencies
        results = await self._execute_with_dependencies(new_tasks)

        for task, result in zip(new_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_task_ids.add(task.id)
                await self._emit(f"[green]Task {task.id} completed[/green]")

        await self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_with_dependencies(self, tasks: list[TaskModel]) -> list[Any]:
        """Execute tasks respecting their dependencies."""
        results: dict[str, Any] = {}
        remaining = {t.id: t for t in tasks}
        
        while remaining:
            # Find tasks whose dependencies are satisfied
            ready = [
                t for t in remaining.values()
                if all(dep in self.completed_task_ids or dep in results for dep in t.dependencies)
            ]
            
            if not ready:
                # Circular dependency or missing deps - execute remaining anyway
                ready = list(remaining.values())
            
            # Run ready tasks in parallel
            task_futures = [
                WorkerAgent(task=task, client=self.client).run()
                for task in ready
            ]
            task_results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            for task, result in zip(ready, task_results):
                results[task.id] = result
                del remaining[task.id]
        
        # Return results in original task order
        return [results.get(t.id) for t in tasks]

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

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
