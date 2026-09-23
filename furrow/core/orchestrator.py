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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, state_file: str | Path | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_file = Path(state_file) if state_file else None
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file and self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                console.print(f"[green]Resumed from cycle {self.cycles} with {len(self.tasks)} tasks[/green]")
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        if not self.state_file:
            return
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
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
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, existing_tasks=self.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing, preserving completed ones
        for new_task in plan.tasks:
            existing = next((t for t in self.tasks if t.id == new_task.id), None)
            if existing and existing.status == "completed":
                new_task.status = "completed"
                new_task.result = existing.result
            self.tasks = [t for t in self.tasks if t.id != new_task.id] + [new_task]

        # Execute tasks respecting dependencies
        await self._execute_tasks()

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

    async def _execute_tasks(self) -> None:
        """Execute tasks in parallel, respecting dependencies."""
        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            return

        # Build dependency graph
        task_map = {t.id: t for t in self.tasks}
        
        # Find tasks with no unmet dependencies
        ready = []
        for task in pending:
            deps_met = all(task_map.get(d, None) and task_map[d].status == "completed" for d in task.dependencies)
            if deps_met:
                ready.append(task)

        if not ready:
            console.print("[yellow]No ready tasks (dependency deadlock). Executing all pending.[/yellow]")
            ready = pending

        with Status(f"[bold yellow]Executing {len(ready)} tasks...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, all_tasks=self.tasks).run()
                for task in ready
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(ready, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        pending = sum(1 for t in self.tasks if t.status == "pending")
        if failed > 0 and pending == 0:
            return False
        if completed >= len(self.tasks) and pending == 0:
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
