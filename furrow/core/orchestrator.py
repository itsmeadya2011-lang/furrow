from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None, 
        settings: Settings | None = None,
        output_callback: Callable[[str], Awaitable[None] | None] | None = None
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient(settings=settings or default_settings)
        self.settings = settings or default_settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[Any] = []  # Track all tasks across cycles
        self.completed_task_summaries: list[str] = []  # For context management
        self.output_callback = output_callback

    async def _print(self, message: str | Panel) -> None:
        """Print to console and optionally send via callback."""
        console.print(message)
        if self.output_callback and isinstance(message, str):
            result = self.output_callback(message)
            if result is not None:
                await result

    async def run(self) -> None:
        await self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            # Check cycle limit
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            
            self.cycles += 1
            await self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Build context with completed task summaries
        context = self._build_context()
        
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, context)
        await self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks for completion detection
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results: list[str | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_summaries = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result  # type: ignore[assignment]
                await self._print(f"[green]Task {task.id} completed[/green]")
                cycle_summaries.append(f"Task {task.id}: {result}")

        # Store summaries for context management
        self.completed_task_summaries.extend(cycle_summaries)
        # Keep only last 20 summaries to prevent context explosion
        if len(self.completed_task_summaries) > 20:
            self.completed_task_summaries = self.completed_task_summaries[-20:]

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._print(f"  • {failure}")
            await self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _build_context(self) -> str:
        """Build context from completed tasks for the planner."""
        if not self.completed_task_summaries:
            return ""
        
        context = "Previous cycles completed:\n"
        for summary in self.completed_task_summaries:
            context += f"  - {summary}\n"
        context += f"\nOriginal goal: {self.original_goal}\n"
        return context

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        if failed > 0:
            return False
        if pending > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
