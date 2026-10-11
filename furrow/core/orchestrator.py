from __future__ import annotations

import asyncio
import json
import logging
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
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.state import StateManager

console = Console()
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_callback: Callable[[str], Awaitable[None]] | None = None,
        resume: bool = False
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_tasks: list[TaskModel] = []
        self.output_callback = output_callback
        self.state_manager = StateManager(self.settings.workspace)
        
        if resume:
            self._resume_state(goal)
        else:
            self.goal = goal
            self.state_manager.save_goal(goal)
            self.state_manager.clear()

    def _resume_state(self, goal: str) -> None:
        """Resume from saved state."""
        saved_goal = self.state_manager.load_goal()
        if saved_goal:
            self.goal = saved_goal
        else:
            self.goal = goal
            self.state_manager.save_goal(goal)
        
        self.cycles = self.state_manager.load_cycle()
        self.all_tasks = self.state_manager.load_tasks()

    async def _emit(self, message: str) -> None:
        """Emit output to both console and callback."""
        console.print(message)
        if self.output_callback:
            try:
                await self.output_callback(message)
            except Exception as e:
                logger.warning(f"Output callback failed: {e}")

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._emit(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            self.state_manager.save_cycle(self.cycles)
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self.state_manager.save_tasks(self.all_tasks)
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        await self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.cycle_tasks = plan.tasks
        self.all_tasks.extend(plan.tasks)

        await self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        await self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Include test failures and relevant context for the next cycle
            failure_context = "\n".join(test_result.failures)
            self.goal = (
                f"Previous goal: {self.goal}\n\n"
                f"Tests failed with the following errors:\n{failure_context}\n\n"
                f"Please fix these issues."
            )
            self.state_manager.save_goal(self.goal)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
