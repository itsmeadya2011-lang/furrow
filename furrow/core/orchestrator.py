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
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient


class Orchestrator:
    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None, 
        console: Console | None = None,
        on_cycle_end: Callable[[], Awaitable[str | None]] | None = None
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()
        self.state_file = Path.cwd() / ".furrow_state.json"
        self._console = console or Console()
        self._on_cycle_end = on_cycle_end
        self._load_state()

    @property
    def console(self) -> Console:
        return self._console

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.completed_task_ids = set(data.get("completed_task_ids", []))
                self.failed_task_ids = set(data.get("failed_task_ids", []))
                # Reconstruct task models
                for t in data.get("tasks", []):
                    task = TaskModel(**t)
                    self.all_tasks.append(task)
                    if task.id in self.completed_task_ids:
                        task.status = "completed"
                    elif task.id in self.failed_task_ids:
                        task.status = "failed"
                self.console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/dim]")
            except Exception as e:
                self.console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "completed_task_ids": list(self.completed_task_ids),
            "failed_task_ids": list(self.failed_task_ids),
            "tasks": [t.model_dump() for t in self.all_tasks],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            # Check max cycles
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self.console.print(f"[yellow]Reached max cycles ({settings.max_cycles}). Halting.[/yellow]")
                break

            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            # Check for mid-loop user input
            if self._on_cycle_end:
                user_input = await self._on_cycle_end()
                if user_input:
                    self.goal = user_input
                    self.console.print(f"[yellow]Goal updated: {user_input}[/yellow]")
            
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_state()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        # Execute tasks respecting dependencies
        results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_task_ids.add(task.id)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_task_ids.add(task.id)
                self.console.print(f"[green]Task {task.id} completed[/green]")

        # Print progress table
        self._print_progress()

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        completed = set()
        results: dict[str, Any] = {}
        
        # Keep executing until all tasks done
        while len(completed) < len(tasks):
            # Find tasks whose dependencies are met
            ready = [
                t for t in tasks 
                if t.id not in completed 
                and all(dep in completed for dep in t.dependencies)
            ]
            
            if not ready:
                # Circular dependency or missing dependency - run remaining
                ready = [t for t in tasks if t.id not in completed]
            
            # Run ready tasks in parallel
            workers = [WorkerAgent(task=task, client=self.client).run() for task in ready]
            worker_results = await asyncio.gather(*workers, return_exceptions=True)
            
            for task, result in zip(ready, worker_results):
                results[task.id] = result
                completed.add(task.id)
        
        return [results[t.id] for t in tasks]

    def _print_progress(self) -> None:
        table = Table(title="Task Progress")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Dependencies", style="dim")
        
        for task in self.all_tasks:
            status_color = {
                "completed": "green",
                "failed": "red",
                "pending": "yellow",
            }.get(task.status, "white")
            
            table.add_row(
                task.id,
                task.description[:60] + ("..." if len(task.description) > 60 else ""),
                f"[{status_color}]{task.status}[/{status_color}]",
                ", ".join(task.dependencies) if task.dependencies else "-",
            )
        self.console.print(table)

    def _is_done(self) -> bool:
        pending = [t for t in self.all_tasks if t.status == "pending"]
        failed = [t for t in self.all_tasks if t.status == "failed"]
        
        if failed:
            return False
        if not pending and self.all_tasks:
            return True
        return False

    def _cleanup_state(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks


# Module-level console for backward compatibility
console = Console()
