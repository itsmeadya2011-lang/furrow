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
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        workspace: Path | None = None,
        max_cycles: int = 0,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.workspace = workspace or Path.cwd()
        self.max_cycles = max_cycles
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()
        self.state_file = self.workspace / ".furrow_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.completed_task_ids = set(data.get("completed", []))
                self.failed_task_ids = set(data.get("failed", []))
                self.cycles = data.get("cycles", 0)
                console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "tasks": [t.model_dump() for t in self.all_tasks],
            "completed": list(self.completed_task_ids),
            "failed": list(self.failed_task_ids),
            "cycles": self.cycles,
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles and self.cycles >= self.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.max_cycles}). Stopping.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.workspace, self.all_tasks)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        new_tasks = [t for t in plan.tasks if t.id not in self.completed_task_ids and t.id not in self.failed_task_ids]
        if not new_tasks:
            console.print("[yellow]All planned tasks already completed or failed.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [WorkerAgent(task=task, client=self.client, workspace=self.workspace).run() for task in new_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(new_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_task_ids.add(task.id)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_task_ids.add(task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

            self.all_tasks.append(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, workspace=self.workspace).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            self.failed_task_ids.clear()

    def _is_done(self) -> bool:
        if self.failed_task_ids:
            return False

        pending = [t for t in self.all_tasks if t.id not in self.completed_task_ids and t.id not in self.failed_task_ids]
        if not pending:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks