from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

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
        status_callback: Callable[[str | dict], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self._status_callback = status_callback

    async def _status(self, message: str | dict) -> None:
        if self._status_callback is not None:
            try:
                await self._status_callback(message)
            except Exception:
                pass
        else:
            console.print(message)

    async def run(self) -> None:
        await self._status(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                await self._status("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break
            await self._status(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._status("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        if self._status_callback is not None:
            try:
                await self._status_callback({"type": "plan", "data": plan.model_dump()})
            except Exception:
                pass
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._status(f"Planning complete: {len(plan.tasks)} tasks")

        if not plan.tasks:
            await self._status("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._status("[bold yellow]Executing tasks in parallel...[/bold yellow]")
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
                await self._status(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._status(f"[green]Task {task.id} completed[/green]")

        await self._status("[bold yellow]Testing...[/bold yellow]")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._status(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._status(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._status(f"  • {failure}")
            await self._status("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.plan.tasks if self.plan else []
