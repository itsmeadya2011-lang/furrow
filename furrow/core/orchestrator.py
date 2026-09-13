from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskStatus
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        stream_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self._stream_callback = stream_callback

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            if max_cycles > 0 and self.cycles > max_cycles:
                console.print(f"[yellow]Reached max cycles ({max_cycles}). Halting.[/yellow]")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._stream(f"Plan: {plan.rationale}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_task(task: Any) -> str:
            async with semaphore:
                agent = WorkerAgent(task=task, client=self.client)
                return await agent.run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = TaskStatus.FAILED
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            self._stream(f"Task {task.id}: {task.status.value}")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            self._stream(f"Test failures: {test_result.failures}")

    def _is_done(self) -> bool:
        if self.plan is None:
            return False
        tasks = self.plan.tasks
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _stream(self, message: str) -> None:
        if self._stream_callback:
            self._stream_callback(message)
