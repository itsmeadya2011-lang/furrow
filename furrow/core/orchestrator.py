from __future__ import annotations

import asyncio
from typing import Any

from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger
from furrow.reporter import ConsoleReporter, Reporter

log = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        reporter: Reporter | None = None,
        max_cycles: int | None = None,
        max_parallel_tasks: int | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.reporter: Reporter = reporter or ConsoleReporter()
        self.max_cycles = max_cycles if max_cycles is not None else self.settings.max_cycles
        self.max_parallel_tasks = max_parallel_tasks or self.settings.max_parallel_tasks
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: Any = None

    async def run(self) -> None:
        self.reporter.print(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self.reporter.print("[yellow]Max cycles reached. Stopping.[/yellow]")
                break
            self.cycles += 1
            self.reporter.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.reporter.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        plan = await self.planner.plan(self.goal)
        self.reporter.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._tasks = plan.tasks

        if not plan.tasks:
            self.reporter.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        sem = asyncio.Semaphore(self.max_parallel_tasks)

        async def _run(task: TaskModel) -> str:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        results = await asyncio.gather(*[_run(t) for t in plan.tasks], return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                self.reporter.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.reporter.print(f"[green]Task {task.id} completed[/green]")

        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._last_test_result = test_result

        if test_result.passed:
            self.reporter.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.reporter.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.reporter.print(f"  • {failure}")
            self.reporter.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self._tasks:
            return True
        if any(t.status == "failed" for t in self._tasks):
            return False
        if any(t.status != "completed" for t in self._tasks):
            return False
        last = self._last_test_result
        return last is None or last.passed

    def _get_tasks(self) -> list[Any]:
        return self._tasks
