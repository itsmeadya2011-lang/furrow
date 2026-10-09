from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient


class OutputCapture:
    """File-like object that captures written text and forwards it to a callback."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        # Flush complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self.callback(stripped)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.callback(self._buffer.strip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.last_tasks: list[Any] = []
        self.last_test_passed: bool = False
        self.on_output = on_output
        if on_output:
            self.console = Console(file=OutputCapture(on_output), force_terminal=False, highlight=False)
        else:
            self.console = Console()

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_tasks = []
            self.last_test_passed = True
            return

        self.last_tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_passed = test_result.passed

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        # Enforce max_cycles if configured
        if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
            return True

        # No tasks planned means we're done
        if not self.last_tasks:
            return True

        # If any task failed, continue to next cycle
        failed = sum(1 for t in self.last_tasks if t.status == "failed")
        if failed > 0:
            return False

        # If not all tasks completed, continue
        completed = sum(1 for t in self.last_tasks if t.status == "completed")
        if completed < len(self.last_tasks):
            return False

        # All tasks completed - halt only if tests passed
        return self.last_test_passed