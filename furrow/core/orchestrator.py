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
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

_console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        emit: Any = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.emit = emit or _console.print
        self.current_plan: Plan | None = None

    async def run(self) -> None:
        self.emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.emit(
                    f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=_console) as status:
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        self.emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=_console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=_console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.emit(f"  • {failure}")
            self.emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.current_plan is None:
            return False
        tasks = self.current_plan.tasks
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.current_plan.tasks if self.current_plan else []
