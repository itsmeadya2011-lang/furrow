from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings
from furrow.core.state import OrchestratorState
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_update: Callable[[str], None] | None = None,
        max_cycles: int = 10,
        state: OrchestratorState | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.on_update = on_update
        self.current_plan = None
        self.test_passed = False
        self.state = state

    def _notify(self, message: str) -> None:
        if self.on_update is not None:
            self.on_update(message)

    async def _save_state(self) -> None:
        """Save current state to file."""
        if self.state is not None:
            self.state.goal = self.goal
            self.state.cycles = self.cycles
            self.state.test_passed = self.test_passed
            if self.current_plan:
                self.state.update_from_plan(self.current_plan)
            await self.state.save()

    @classmethod
    async def from_state(
        cls,
        state: OrchestratorState,
        client: LLMClient | None = None,
    ) -> Orchestrator:
        """Resume orchestrator from saved state."""
        data = await state.load()
        orch = cls(goal=data.get("goal", ""), client=client)
        orch.cycles = data.get("cycles", 0)
        orch.test_passed = data.get("test_passed", False)
        orch.state = state
        return orch

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._notify("Planning...")
        while True:
            self.cycles += 1
            self._notify(f"Cycle {self.cycles} started")
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            effective_max = settings.max_cycles if settings.max_cycles > 0 else self.max_cycles
            if self.cycles >= effective_max:
                console.print(f"[bold red]Max cycles ({effective_max}) reached. Halting.[/bold red]")
                self._notify("Max cycles reached")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._notify("Goal complete")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
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
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._notify(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed: {result}[/green]")
                self._notify(f"Task {task.id} completed: {result}")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.test_passed = True
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._notify(f"Tests passed: {test_result.summary}")
        else:
            self.test_passed = False
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            self._notify(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        if self.state is not None:
            self.state.add_history_entry(self.cycles, test_result)
            await self._save_state()

    def _is_done(self) -> bool:
        if not self._get_tasks():
            return True
        return self.test_passed

    def _get_tasks(self) -> list[Any]:
        if self.current_plan:
            return self.current_plan.tasks
        return []
