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
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.completed_tasks: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        state_path = Path(self.settings.workspace) / STATE_FILE
        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = json.load(f)
                    self.goal = data.get("goal", self.goal)
                    self.cycles = data.get("cycles", 0)
                    self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                    self.completed_tasks = [TaskModel(**t) for t in data.get("completed_tasks", [])]
                console.print(f"[dim]Loaded state from {state_path}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        state_path = Path(self.settings.workspace) / STATE_FILE
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "completed_tasks": [t.model_dump() for t in self.completed_tasks],
        }
        try:
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_state()
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        # Filter out completed tasks from current tasks
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        
        # If no pending tasks, plan new ones
        if not pending_tasks:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal, self._get_project_context())
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return

            # Add new tasks to the task list
            for task in plan.tasks:
                task.status = "pending"
            self.tasks.extend(plan.tasks)
            pending_tasks = plan.tasks

        # Execute pending tasks in parallel (respecting dependencies)
        await self._execute_tasks(pending_tasks)

        # Run tests
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to include fixing failures
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self._save_state()
        self._print_task_summary()

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        # Group tasks by dependency level
        ready_tasks = [t for t in tasks if self._dependencies_met(t)]
        if not ready_tasks:
            console.print("[yellow]No tasks ready to execute (dependencies not met)[/yellow]")
            return

        with Status(f"[bold yellow]Executing {len(ready_tasks)} tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in ready_tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(ready_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

    def _dependencies_met(self, task: TaskModel) -> bool:
        for dep_id in task.dependencies:
            dep_task = next((t for t in self.tasks if t.id == dep_id), None)
            if not dep_task or dep_task.status != "completed":
                return False
        return True

    def _get_project_context(self) -> str:
        """Get project context for planning."""
        workspace = Path(self.settings.workspace)
        files = []
        for ext in ["*.py", "*.js", "*.ts", "*.json", "*.md", "*.yaml", "*.yml", "*.toml"]:
            files.extend(str(f.relative_to(workspace)) for f in workspace.rglob(ext) if f.is_file())
        return f"Project files:\n{chr(10).join(sorted(files)[:50])}"

    def _is_done(self) -> bool:
        all_tasks = self.tasks + self.completed_tasks
        if not all_tasks:
            return False
        completed = sum(1 for t in all_tasks if t.status == "completed")
        failed = sum(1 for t in all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(all_tasks):
            return True
        return False

    def _print_task_summary(self) -> None:
        table = Table(title="Task Summary")
        table.add_column("ID")
        table.add_column("Description")
        table.add_column("Status")
        table.add_column("Files")
        for task in self.tasks + self.completed_tasks:
            table.add_row(task.id, task.description[:50], task.status, ", ".join(task.files[:3]))
        console.print(table)

    def _cleanup_state(self) -> None:
        state_path = Path(self.settings.workspace) / STATE_FILE
        if state_path.exists():
            state_path.unlink()