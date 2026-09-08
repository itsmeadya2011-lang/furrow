from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self._on_output = on_output

    def _ws_send(self, text: str) -> None:
        if self._on_output is not None:
            self._on_output(text)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        self._ws_send(f"Furrow\nGoal: {self.original_goal}")
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._ws_send(f"\n═══ Cycle {self.cycles} ═══")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._ws_send("Goal complete. Halting.")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[bold yellow]Reached max_cycles limit ({self.settings.max_cycles}). Halting.[/bold yellow]")
                self._ws_send(f"Reached max_cycles limit ({self.settings.max_cycles}). Halting.")
                break

    async def _cycle(self) -> None:
        feedback = self.goal if self.cycles > 1 else None
        planning_goal = self.original_goal
        if feedback:
            planning_goal = f"{self.original_goal}\n\nPrevious feedback:\n{feedback}"
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(planning_goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._ws_send(f"Plan: {plan.model_dump()}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._ws_send("No tasks planned. Goal may be complete.")
            return

        self._ws_send(f"Executing {len(plan.tasks)} tasks in parallel...")
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
                self._ws_send(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._ws_send(f"Task {task.id} completed")

        self._ws_send("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.original_goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._ws_send(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            self._ws_send(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                self._ws_send(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._ws_send("Will attempt fix in next cycle.")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
