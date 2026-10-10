from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)
console = Console()

OutputCallback = Callable[[str], None | Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.output_callback = output_callback
        self._plain_console = Console(force_terminal=False, no_color=True)

    async def _emit(self, text: str) -> None:
        console.print(text)
        if self.output_callback:
            plain = self._plain_console.render_str(text).plain
            result = self.output_callback(plain)
            if inspect.iscoroutine(result):
                await result

    async def run(self) -> None:
        goal_panel = Panel.fit(
            f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
            title="Furrow",
        )
        await self._emit(goal_panel)
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            if max_cycles > 0:
                await self._emit(
                    f"\n[bold cyan]═══ Cycle {self.cycles}/{max_cycles} ═══[/bold cyan]"
                )
            else:
                await self._emit(
                    f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
                )
            await self._cycle()
            if self._is_done():
                logger.info("goal_complete")
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                logger.warning("max_cycles_reached")
                await self._emit(
                    f"[bold yellow]Reached max_cycles limit ({max_cycles}). Halting.[/bold yellow]"
                )
                break

    async def _cycle(self) -> None:
        logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
        await self._emit("Planning...")
        plan = await self.planner.plan(self.goal)
        logger.info("plan_created", task_count=len(plan.tasks))
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks into self.tasks, updating existing ones by id.
        existing_by_id = {t.id: t for t in self.tasks}
        for task in plan.tasks:
            if task.id in existing_by_id:
                existing = existing_by_id[task.id]
                existing.status = task.status
                existing.result = task.result
            else:
                self.tasks.append(task)
                existing_by_id[task.id] = task

        await self._emit("Executing tasks in parallel...")
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        completed = sum(1 for r in results if not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))
        logger.info("tasks_executed", completed=completed, failed=failed)

        await self._emit("Testing...")
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        logger.info("tests_result", passed=test_result.passed, summary=test_result.summary)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            new_goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            await self._emit(f"[yellow]New goal:[/yellow] {new_goal}")
            self.goal = new_goal

        # Summary of all tasks so far.
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        total = len(self.tasks)
        await self._emit(
            f"[bold]Task summary[/bold]: {completed} completed, {failed} failed, {total} total"
        )

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        for t in tasks:
            if t.status == "failed":
                return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks