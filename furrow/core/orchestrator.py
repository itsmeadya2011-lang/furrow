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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Any = None,
    ) -> None:
        self.goal = goal
        self._original_goal = goal
        self._fixes: list[str] = []
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._on_event = on_event

    def _emit(self, message: str) -> None:
        if self._on_event:
            try:
                self._on_event(message)
            except Exception:
                pass
        console.print(message)

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    def _effective_goal(self) -> str:
        if self._fixes:
            fixes = "\n".join(f"- {f}" for f in self._fixes)
            return f"Fix failing tests:\n{fixes}\n\nOriginal goal: {self._original_goal}"
        return self._original_goal

    async def _cycle(self) -> None:
        project_files = ""
        try:
            workspace = self.client.settings.workspace
            project_files = self.client.list_files(str(workspace))
        except Exception as e:
            self._emit(f"[yellow]Could not list project files: {e}[/yellow]")

        goal_for_planner = self._effective_goal()

        with Status("[bold yellow]Planning...", console=console) as status:
            try:
                plan = await self.planner.plan(goal_for_planner, project_context=project_files)
            except Exception as e:
                self._emit(f"[red]Planning failed: {e}[/red]")
                self._current_plan = None
                return

        self._current_plan = plan
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self._original_goal, plan.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            self._fixes = []
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  - {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            for failure in test_result.failures:
                self._fixes.append(failure)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        if all(t.status == "completed" for t in tasks):
            return True
        max_cycles = getattr(self.client.settings, "max_cycles", 0) or 0
        if max_cycles > 0 and self.cycles >= max_cycles:
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        if self._current_plan is None:
            return []
        return self._current_plan.tasks