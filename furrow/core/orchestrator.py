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
    def __init__(self, goal: str, client: LLMClient | None = None, state_file: Path | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.state_file = state_file or (settings.workspace / ".furrow_state.json")
        self.all_tasks: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.all_tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            console.print(f"[yellow]Could not save state: {e}[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done() or (settings.max_cycles > 0 and self.cycles >= settings.max_cycles):
                if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                    console.print(f"[bold yellow]Max cycles ({settings.max_cycles}) reached.[/bold yellow]")
                else:
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed tasks
        pending_tasks = [t for t in plan.tasks if not self._is_task_completed(t.id)]
        if not pending_tasks:
            console.print("[green]All tasks completed.[/green]")
            return

        # Limit parallel tasks
        max_parallel = settings.max_parallel_tasks
        for i in range(0, len(pending_tasks), max_parallel):
            batch = pending_tasks[i:i + max_parallel]
            with Status(f"[bold yellow]Executing {len(batch)} tasks in parallel...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                # Update or add to all_tasks
                self._update_task(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Add test failures as new tasks for next cycle
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_task_completed(self, task_id: str) -> bool:
        for t in self.all_tasks:
            if t.id == task_id and t.status == "completed":
                return True
        return False

    def _update_task(self, task: TaskModel) -> None:
        for i, t in enumerate(self.all_tasks):
            if t.id == task.id:
                self.all_tasks[i] = task
                return
        self.all_tasks.append(task)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        if failed > 0:
            return False
        if pending == 0 and completed >= len(self.all_tasks):
            return True
        return False
