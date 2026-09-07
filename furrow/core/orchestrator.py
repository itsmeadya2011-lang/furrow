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
from furrow.config import Plan, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._goal_complete = False
        self._cycle_errors = 0
        self._output_callback = output_callback

    async def _emit(self, renderable: Any) -> None:
        if self._output_callback is None:
            return
        plain = Console(force_terminal=False, color_system=None, width=console.width)
        with plain.capture() as cap:
            plain.print(renderable)
        text = cap.get().strip()
        if text:
            await self._output_callback(text)

    async def run(self) -> None:
        self._emit(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if 0 < settings.max_cycles <= self.cycles:
                self._emit(
                    f"[yellow]Reached max_cycles={settings.max_cycles}. Halting.[/yellow]"
                )
                break
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
            self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                self._emit("[yellow]No tasks planned.[/yellow]")
                # Run tests once with no new tasks to confirm completion
                with Status("[bold yellow]Testing...", console=console) as status:
                    test_result = await TesterAgent(client=self.client).run(
                        self.goal, self.tasks
                    )
                if test_result.passed:
                    self._goal_complete = True
                else:
                    self._emit(
                        f"[red]Tests failed with no new tasks: {test_result.summary}[/red]"
                    )
                    for failure in test_result.failures:
                        self._emit(f"  • {failure}")
                    self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
                    self._cycle_errors += 1
                return

            self.tasks = plan.tasks

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
                    self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self._emit(f"[green]Task {task.id} completed[/green]")

            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await TesterAgent(client=self.client).run(
                    self.goal, plan.tasks
                )

            if test_result.passed:
                self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
                self._goal_complete = True
            else:
                self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    self._emit(f"  • {failure}")
                self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
                self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

            self._cycle_errors = 0
        except Exception as e:
            self._cycle_errors += 1
            self._emit(f"[red]Cycle {self.cycles} error: {e}[/red]")
            if self._cycle_errors >= 3:
                self._emit(
                    f"[red]Aborting after {self._cycle_errors} consecutive cycle errors.[/red]"
                )
                self._goal_complete = True

    def _is_done(self) -> bool:
        return self._goal_complete

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
