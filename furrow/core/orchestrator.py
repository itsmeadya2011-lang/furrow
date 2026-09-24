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
from furrow.config import Plan, TestResult, Settings, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, config: Settings = settings) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.config = config
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Get project context for planner
        project_files = self.client.list_files(self.config.workspace)
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, project_files, self.all_tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking
        for task in plan.tasks:
            task.status = "pending"
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, config=self.config).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_results = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                cycle_results.append({"task_id": task.id, "status": "failed", "result": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                cycle_results.append({"task_id": task.id, "status": "completed", "result": result})

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "results": cycle_results,
            "test_result": test_result.model_dump(),
        })

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

    def get_status(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self.all_tasks],
            "history": self.cycle_history,
        }