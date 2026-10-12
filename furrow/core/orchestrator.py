from __future__ import annotations

import asyncio
import inspect
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
from furrow.config import Plan, TestResult, TaskModel, settings
from furrow.core.state import StateManager
from furrow.llm import LLMClient


OutputCallback = Callable[[str], Awaitable[None]] | Callable[[str], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        on_output: OutputCallback | None = None,
        resume: bool = False,
        state_dir: Path | None = None
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.max_cycles = max_cycles
        self.on_output = on_output
        self._console = Console()
        self._state_dir = state_dir or settings.workspace
        self._state = StateManager(self._state_dir)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.plan: Plan | None = None

        if resume:
            self._load_state()
        else:
            self._state.clear()

    def _load_state(self) -> None:
        """Load state from disk."""
        state = self._state.load()
        self.cycles = state.get("cycles", 0)
        self.goal = state.get("goal", self.goal)
        self.all_tasks = self._state.get_tasks()
        self.plan = self._state.get_plan()
        self._emit(f"[cyan]Resumed from cycle {self.cycles}[/cyan]")

    def _save_state(self) -> None:
        """Save current state to disk."""
        self._state.set_cycles(self.cycles)
        self._state.set_goal(self.goal)
        self._state.set_tasks(self.all_tasks)
        if self.plan:
            self._state.set_plan(self.plan)

    def _emit(self, message: str) -> None:
        """Emit a message to the output callback or console."""
        if self.on_output:
            # Handle both sync and async callbacks
            if inspect.iscoroutinefunction(self.on_output):
                asyncio.create_task(self.on_output(message))
            else:
                self.on_output(message)
        else:
            self._console.print(message)

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self._emit(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        self.plan = plan
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        # Filter tasks that are ready to run (dependencies met)
        ready_tasks = self._get_ready_tasks(plan.tasks)
        if not ready_tasks:
            self._emit("[yellow]No tasks ready to run (waiting on dependencies).[/yellow]")
            return

        self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in ready_tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(ready_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_ready_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Return tasks whose dependencies are all completed."""
        completed_ids = {t.id for t in self.all_tasks if t.status == "completed"}
        ready = []
        for task in tasks:
            if task.status in ("pending", "failed"):
                if all(dep in completed_ids for dep in task.dependencies):
                    ready.append(task)
        return ready

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
