from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

console = Console()

EventCallback = Callable[[str], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._no_tasks = False
        self._semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        self._on_event = on_event

    async def _emit(self, message: str) -> None:
        if self._on_event is not None:
            await self._on_event(message)
        else:
            console.print(message)

    async def run(self) -> None:
        goal_text = f"[bold green]Furrow[/bold green]\nGoal: {self.goal}"
        console.print(Panel.fit(goal_text, title="Furrow"))
        await self._emit(f"Furrow — Goal: {self.goal}")

        while True:
            self.cycles += 1
            await self._emit(f"\n═══ Cycle {self.cycles} ═══")
            await self._cycle()

            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                limit = self.settings.max_cycles
                await self._emit(f"[yellow]Reached max_cycles ({limit}). Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(
            "Planned "
            + str(len(plan.tasks))
            + " task(s): "
            + "; ".join(f"{t.id}: {t.description}" for t in plan.tasks)
        )

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._no_tasks = True
            return

        self._record_tasks(plan.tasks)

        async def _run(task: TaskModel) -> tuple[TaskModel, str | Exception]:
            async with self._semaphore:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    return task, result
                except Exception as e:
                    return task, e

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(*[_run(task) for task in plan.tasks])

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = (
                f"{self.original_goal}\n\nFix the following failing tests:\n"
                + "\n".join(test_result.failures)
            )

    def _record_tasks(self, tasks: list[TaskModel]) -> None:
        task_map = {t.id: t for t in self.tasks}
        for task in tasks:
            task_map[task.id] = task
        self.tasks = list(task_map.values())

    def _is_done(self) -> bool:
        # The loop halts only when the planner declares the goal complete
        # (returns no tasks). Failed tasks keep the loop running so the next
        # cycle can attempt fixes; max_cycles is the safety valve.
        return self._no_tasks

    def _get_tasks(self) -> list[TaskModel]:
        return list(self.tasks)