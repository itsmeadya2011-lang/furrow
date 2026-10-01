from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import structlog
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
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_handler: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._done = False
        self._all_failed = False
        self._on_output: Callable[[str], Any] = output_handler or (lambda _: None)

    async def _emit(self, msg: str) -> None:
        """Send an output message to the handler.

        Supports both synchronous and asynchronous handlers.
        """
        result = self._on_output(msg)
        if asyncio.iscoroutine(result):
            await result

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit(f"Furrow started — goal: {self.goal}\n")
        max_cycles = self.client.settings.max_cycles
        while not self._done:
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print("[yellow]Max cycles reached. Stopping.[/yellow]")
                await self._emit("Max cycles reached. Stopping.\n")
                logger.warning("max_cycles_reached", cycles=self.cycles, max_cycles=max_cycles)
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(f"Cycle {self.cycles}\n")
            await self._cycle()
            if self._is_done():
                if self._done:
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    await self._emit("Goal complete.\n")
                    logger.info("goal_complete", cycles=self.cycles)
                else:
                    console.print("[bold red]All tasks failed. Halting.[/bold red]")
                    await self._emit("All tasks failed. Halting.\n")
                    logger.warning("all_tasks_failed", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale}\n")
        await self._emit(f"Tasks: {[t.id for t in plan.tasks]}\n")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("No tasks planned. Treating as complete.\n")
            self._done = True
            return

        self.tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=t, client=self.client).run()
                for t in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(f"Task {task.id}: FAILED — {result}\n")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result if isinstance(result, str) else str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"Task {task.id}: COMPLETED\n")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}\n")
            self._done = True
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            await self._emit(f"Tests failed: {test_result.summary}\n")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}\n")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.info("tests_failed_retry", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self._done:
            return True
        # If no tasks were planned/executed this cycle, nothing to do.
        if not self.tasks:
            return True
        # If all tasks failed, there's no progress to build on — halt.
        if all(t.status == "failed" for t in self.tasks):
            self._all_failed = True
            return True
        # If some tasks failed, retry in the next cycle.
        if any(t.status == "failed" for t in self.tasks):
            return False
        # All tasks completed but tests haven't passed yet (tests failed or
        # haven't run). Keep cycling to fix issues.
        return False
