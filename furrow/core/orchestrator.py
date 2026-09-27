from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_cycle: Callable[[int, str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_passed: bool | None = None
        self.on_cycle = on_cycle

    async def run(self) -> None:
        self.client.validate()
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            summary = "No tasks planned. Goal may be complete."
            console.print(f"[yellow]{summary}[/yellow]")
            await self._emit_cycle(summary)
            return

        existing = {t.id: t for t in self._tasks}
        for task in plan.tasks:
            existing[task.id] = task
        self._tasks = list(existing.values())

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(run_task(t) for t in plan.tasks), return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._last_test_passed = test_result.passed

        if test_result.passed:
            summary = f"Tests passed: {test_result.summary}"
            console.print(f"[green]{summary}[/green]")
        else:
            summary = f"Tests failed: {test_result.summary}"
            console.print(f"[red]{summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        await self._emit_cycle(summary)

    async def _emit_cycle(self, summary: str) -> None:
        if self.on_cycle is None:
            return
        try:
            await self.on_cycle(self.cycles, summary)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]on_cycle callback error: {e}[/red]")

    def _is_done(self) -> bool:
        if self._last_test_passed is False:
            return False
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self._get_tasks())

    def _get_tasks(self) -> list[Any]:
        return self._tasks
