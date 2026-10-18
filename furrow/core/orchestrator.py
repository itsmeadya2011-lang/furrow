from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        on_output: Callable[[str], None] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings or Settings()
        self.on_output = on_output
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self._original_goal: str = goal

    def _emit(self, text: str) -> None:
        if self.on_output is not None:
            callback_text = text
            if hasattr(text, "__rich_console__"):
                # Rich renderable (e.g. Panel): capture plain text for the callback.
                buffer = StringIO()
                cap_console = Console(file=buffer, color_system=None, width=100)
                cap_console.print(text)
                callback_text = buffer.getvalue()
            self.on_output(callback_text)
        console.print(text)

    async def run(self) -> None:
        self._original_goal = self.goal
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._emit(
                    f"[yellow]Reached max_cycles limit ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                break

        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        self._emit(
            f"[bold]Done.[/bold] Cycles: {self.cycles} | Completed: {completed} | Failed: {failed}"
        )

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan: Plan = await self.planner.plan(self.goal)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks

        semaphore: asyncio.Semaphore | None = None
        if self.settings.max_parallel_tasks > 0:
            semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_task(task: Any) -> Any:
            agent = WorkerAgent(task=task, client=self.client)
            if semaphore is None:
                return await agent.run()
            async with semaphore:
                return await agent.run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(*(_run_task(t) for t in self.tasks), return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result: TestResult = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"{self._original_goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed >= len(self.tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks