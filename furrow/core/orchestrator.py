from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from .logging import get_logger

console = Console()
_text_console = Console(force_terminal=False, no_color=True)


def _to_text(renderable: Any) -> str:
    with _text_console.capture() as cap:
        _text_console.print(renderable, soft_wrap=True)
    return cap.get().rstrip()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client, settings=self.client.settings)
        self.worker = WorkerAgent(client=self.client, settings=self.client.settings)
        self.tester = TesterAgent(client=self.client, settings=self.client.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.max_cycles = self.client.settings.max_cycles if self.client else 0
        self.output_callback = output_callback

        self.logger = get_logger(__name__)

    async def _out(self, renderable: Any) -> None:
        console.print(renderable)
        if self.output_callback is not None:
            result = self.output_callback(_to_text(renderable))
            if inspect.isawaitable(result):
                await result

    async def run(self) -> None:
        await self._out(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self.logger.info("Starting Furrow orchestrator", goal=self.goal)
        while True:
            self.cycles += 1
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self._out("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break
            await self._out(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._out("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        self.logger.info("Starting planning phase")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.logger.debug("Plan generated", plan=plan.model_dump())
        await self._out(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._out("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store tasks from this cycle (append new tasks)
        for task in plan.tasks:
            # Avoid duplicating tasks if they already exist
            if not any(t.id == task.id for t in self.tasks):
                self.tasks.append(task)
            else:
                # Update existing task if re-planned
                for i, existing in enumerate(self.tasks):
                    if existing.id == task.id:
                        self.tasks[i] = task
                        break

        self.logger.info("Starting execution phase", num_tasks=len(plan.tasks))
        with Status("[bold yellow]Executing tasks...", console=console):
            # Filter tasks that haven't been planned yet? Wait - we already have plan.tasks
            # Separate by dependencies
            completed_task_ids = {t.id for t in self.tasks if t.status == "completed"}
            runnable_tasks = []
            skipped_tasks = []

            for task in plan.tasks:
                if not task.dependencies:
                    runnable_tasks.append(task)
                else:
                    deps_met = all(dep in completed_task_ids for dep in task.dependencies)
                    if deps_met:
                        runnable_tasks.append(task)
                    else:
                        skipped_tasks.append(task)
                        task.status = "pending"
                        await self._out(f"[yellow]Task {task.id} skipped (unmet dependencies): {task.dependencies}[/yellow]")

            if not runnable_tasks:
                await self._out("[yellow]No runnable tasks this cycle.[/yellow]")
                return

            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.client.settings).run()
                for task in runnable_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(runnable_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._out(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._out(f"[green]Task {task.id} completed[/green]")

        self.logger.info("Execution phase completed")

        self.logger.info("Starting testing phase")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.client.settings).run(self.goal, self.tasks)

        if test_result.passed:
            await self._out(f"[green]Tests passed: {test_result.summary}[/green]")
            self.logger.info("Testing phase completed", passed=True)
        else:
            await self._out(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._out(f"  • {failure}")
            await self._out("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            # Mark tasks that need rework if they relate to failures
            # This keeps track of problematic tasks
            for task in self.tasks:
                if task.status == "completed" and any(task.id in str(failure) or task.description in str(failure) for failure in test_result.failures):
                    task.status = "pending"
            self.logger.info("Testing phase completed", passed=False)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        if completed >= len(self.tasks):
            return True
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            return True
        return False