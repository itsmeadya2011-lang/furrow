from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import aiofiles
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
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = Path.cwd() / ".furrow_state.json"

    async def run(self) -> None:
        await self._load_state()
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            await self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to tracking
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

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

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to focus on fixing failures
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

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

    async def _save_state(self) -> None:
        """Save orchestrator state to disk."""
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.all_tasks],
        }
        try:
            async with aiofiles.open(self.state_file, "w") as f:
                await f.write(json.dumps(state, indent=2))
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save state: {e}[/yellow]")

    async def _load_state(self) -> None:
        """Load orchestrator state from disk."""
        if not self.state_file.exists():
            return
        try:
            async with aiofiles.open(self.state_file, "r") as f:
                content = await f.read()
            state = json.loads(content)
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            self.all_tasks = [TaskModel(**t) for t in state.get("tasks", [])]
            if self.all_tasks:
                console.print(f"[green]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load state: {e}[/yellow]")
