from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings or Settings()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.console = console or Console()

    async def run(self) -> None:
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            log.info("Cycle starting", cycle=self.cycles, goal=self.goal)
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                log.info("Goal complete", cycle=self.cycles)
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        log.info("Planning started", cycle=self.cycles, goal=self.goal)
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        log.info("Planning completed", cycle=self.cycles, num_tasks=len(plan.tasks))
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.tasks = plan.tasks

        if not plan.tasks:
            log.info("No tasks planned", cycle=self.cycles)
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        log.info("Executing tasks", cycle=self.cycles, num_tasks=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [WorkerAgent(task=task, client=self.client).run() for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                log.error("Task failed", cycle=self.cycles, task_id=task.id, error=str(result))
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                log.info("Task completed", cycle=self.cycles, task_id=task.id)
                self.console.print(f"[green]Task {task.id} completed[/green]")

        log.info("Testing started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            log.info("Tests passed", cycle=self.cycles, summary=test_result.summary)
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            log.warning(
                "Tests failed",
                cycle=self.cycles,
                summary=test_result.summary,
                num_failures=len(test_result.failures),
            )
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        if failed > 0:
            return False
        if len(tasks) == 0:
            return True
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks
