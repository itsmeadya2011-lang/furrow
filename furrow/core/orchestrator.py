from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings if settings is not None else Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.test_result: TestResult | None = None
        self._output_callback = output_callback
        self._console = Console()

    async def run(self) -> None:
        self._log_header()
        while not self._is_done():
            self.cycles += 1
            self._console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self._output_callback:
                self._output_callback(f"Cycle {self.cycles}")
            await self._cycle()
        # final summary
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        self._console.print(Panel.fit(
            f"[bold]Furrow finished[/bold] after {self.cycles} cycle(s).\n"
            f"Completed: {completed}  Failed: {failed}",
            title="Summary",
            border_style="cyan",
        ))
        if self._output_callback is not None:
            status = "unknown"
            if self.test_result is not None:
                status = "passed" if self.test_result.passed else "failed"
            self._output_callback(f"Furrow finished {self.cycles} cycle(s). Final status: {status}")

    def _log_header(self) -> None:
        header = f"[bold green]Furrow[/bold green]\nGoal: {self.goal}"
        self._console.print(Panel.fit(header, title="Furrow"))
        if self._output_callback:
            self._output_callback(f"Furrow starting. Goal: {self.goal}")

    async def _cycle(self) -> None:
        plan = await self.planner.plan(self.goal)
        self._console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if self._output_callback:
            self._output_callback(f"Plan: {plan.rationale}")

        # replace tracked tasks with this cycle's plan tasks (re-plans on retry)
        self.tasks = list(plan.tasks)
        if not plan.tasks:
            if self._output_callback:
                self._output_callback("No tasks planned. Goal may be complete.")
            return

        results = await asyncio.gather(
            *[WorkerAgent(task=t, client=self.client).run() for t in self.tasks],
            return_exceptions=True,
        )
        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._console.print(f"[red]Task {task.id} failed: {result}[/red]")
                if self._output_callback:
                    self._output_callback(f"Task {task.id} FAILED: {result}")
            else:
                task.status = "completed"
                task.result = result
                self._console.print(f"[green]Task {task.id} completed[/green]")
                if self._output_callback:
                    self._output_callback(f"Task {task.id} completed: {result}")

        self.test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        if self.test_result.passed:
            self._console.print(f"[green]Tests passed: {self.test_result.summary}[/green]")
            if self._output_callback:
                self._output_callback(f"Tests PASSED: {self.test_result.summary}")
        else:
            self._console.print(f"[red]Tests failed: {self.test_result.summary}[/red]")
            for failure in self.test_result.failures:
                self._console.print(f"  • {failure}")
            self._console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            if self._output_callback:
                self._output_callback("Tests FAILED; will retry.")
            self.goal = "Fix failing tests:\n" + "\n".join(self.test_result.failures)

    def _is_done(self) -> bool:
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            self._console.print("[yellow]Max cycles reached. Stopping.[/yellow]")
            if self._output_callback:
                self._output_callback("Max cycles reached. Stopping.")
            return True
        if self.test_result is not None and self.test_result.passed:
            self._console.print("[bold green]Goal complete. Halting.[/bold green]")
            if self._output_callback:
                self._output_callback("Goal complete.")
            return True
        # No tasks were planned this cycle (after at least one cycle ran) →
        # nothing left to attempt. This also stops the loop when the planner can
        # no longer produce fix tasks, avoiding an infinite loop on unfixable goals.
        if not self.tasks and self.cycles > 0:
            if self._output_callback:
                self._output_callback("No more tasks to plan. Stopping.")
            return True
        # Tests failed but tasks exist → retry with a fresh plan
        if self.test_result is not None and not self.test_result.passed:
            return False
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
