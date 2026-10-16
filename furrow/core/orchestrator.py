from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TestResult
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self._tasks: list[Any] = []
        self._last_test_result: TestResult | None = None

    async def run(self) -> None:
        log.info("Furrow starting", goal=self.goal, max_cycles=self.max_cycles)
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("Furrow halted", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("Plan created", task_count=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [WorkerAgent(task=task, client=self.client).run() for task in plan.tasks]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("Task failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                log.info("Task completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("Tests passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            log.warning("Tests failed, will retry", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            console.print(
                f"[bold yellow]Reached max cycles ({self.max_cycles}). Halting.[/bold yellow]"
            )
            return True

        if not self._tasks:
            return False

        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")

        return failed == 0 and completed >= len(self._tasks)

    def _get_tasks(self) -> list[Any]:
        return self._tasks
