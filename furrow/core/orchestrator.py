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

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_file = state_file or (settings.workspace / ".furrow_state.json")
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                console.print(f"[dim]Loaded state: {len(self.tasks)} tasks, cycle {self.cycles}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if settings.max_cycles and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({settings.max_cycles}). Halting.[/yellow]")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Update task list with new tasks from plan (merge with existing)
        for new_task in plan.tasks:
            existing = next((t for t in self.tasks if t.id == new_task.id), None)
            if existing:
                # Update existing task with new plan info but preserve status/result
                existing.description = new_task.description
                existing.files = new_task.files
                existing.dependencies = new_task.dependencies
            else:
                self.tasks.append(new_task)

        # Filter tasks to only those not yet completed
        pending_tasks = [t for t in self.tasks if t.status != "completed"]
        
        if not pending_tasks:
            console.print("[green]All tasks completed![/green]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
