from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._plan: Plan | None = None
        self.logger = structlog.get_logger()

    async def run(self) -> None:
        self.logger.info("run_start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self.logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            await self._cycle()

            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                self.logger.info(
                    "max_cycles_reached",
                    cycles=self.cycles,
                    max_cycles=settings.max_cycles,
                )
                break

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self.logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        self.logger.info("planning", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.logger.info("planning_complete", tasks=len(plan.tasks))

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
                self.logger.info("task_execution_result", task_id=task.id, status="failed")
            else:
                task.status = "completed"
                task.result = str(result) if isinstance(result, str) else json.dumps(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                if isinstance(result, dict):
                    files_written = result.get("files_written", [])
                    if files_written:
                        console.print(f"[dim]  Written: {', '.join(files_written)}[/dim]")
                self.logger.info("task_execution_result", task_id=task.id, status="completed")

        self.logger.info("testing", goal=self.goal)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.logger.info("testing_complete", passed=True)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.logger.info("testing_complete", passed=False, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self.logger.info("cycle_end", cycle=self.cycles)

    def _is_done(self) -> bool:
        if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
            return True
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._plan.tasks if self._plan else []

    @property
    def summary(self) -> dict[str, Any]:
        tasks = self._get_tasks()
        return {
            "cycles_completed": self.cycles,
            "tasks_completed": sum(1 for t in tasks if t.status == "completed"),
            "tasks_failed": sum(1 for t in tasks if t.status == "failed"),
        }
