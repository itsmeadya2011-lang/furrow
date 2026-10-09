from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()
settings = Settings()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        on_output: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.max_cycles = max_cycles if max_cycles is not None else settings.max_cycles
        self._on_output = on_output
        self._stopped = False

    def set_console_callback(self, callback: Optional[Callable[[str], Any]]) -> None:
        """Set a callback that receives output strings instead of (or in addition to) console printing."""
        self._on_output = callback

    def stop(self) -> None:
        """Signal the orchestrator to stop at the next cycle boundary."""
        self._stopped = True

    def _emit(self, text: str) -> None:
        """Emit text to the callback if set, otherwise print to the console."""
        if self._on_output is not None:
            result = self._on_output(text)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
        else:
            console.print(text)

    async def run(self, max_cycles: int | None = None) -> None:
        if max_cycles is not None:
            self.max_cycles = max_cycles
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while not self._stopped:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self._emit(f"[bold yellow]Max cycles ({self.max_cycles}) reached. Halting.[/bold yellow]")
                break
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks into self.tasks, avoiding duplicates by id
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)
                existing_ids.add(task.id)

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
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
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