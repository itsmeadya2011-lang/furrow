from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.on_progress = on_progress
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None

    def _notify(self, message: str) -> None:
        if self.on_progress:
            self.on_progress(message)
        console.print(message)

    async def run(self) -> None:
        self._notify(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            self._notify(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._notify("[bold green]Goal complete. Halting.[/bold green]")
                break
            if (
                self.client.settings.max_cycles > 0
                and self.cycles >= self.client.settings.max_cycles
            ):
                self._notify(
                    f"[yellow]Reached max_cycles "
                    f"({self.client.settings.max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._notify(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._tasks = plan.tasks

        if not plan.tasks:
            self._notify("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute_tasks()

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)
        self._last_test_result = test_result

        if test_result.passed:
            self._notify(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._notify(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._notify(f"  • {failure}")
            self._notify("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self) -> None:
        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel) if max_parallel > 0 else None

        async def _run_task(task: TaskModel) -> None:
            async def _inner() -> None:
                worker = WorkerAgent(task=task, client=self.client)
                try:
                    result = await worker.run()
                    task.status = "completed"
                    task.result = result
                    self._notify(f"[green]Task {task.id} completed[/green]")
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    self._notify(f"[red]Task {task.id} failed: {e}[/red]")

            if semaphore is not None:
                async with semaphore:
                    await _inner()
            else:
                await _inner()

        completed: set[str] = set()
        remaining: set[TaskModel] = set(self._tasks)

        while remaining:
            wave = [
                t for t in remaining
                if all(dep in completed for dep in t.dependencies)
            ]
            if not wave:
                wave = list(remaining)

            await asyncio.gather(*[_run_task(t) for t in wave])

            for t in wave:
                remaining.discard(t)
                if t.status == "completed":
                    completed.add(t.id)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        all_completed = all(t.status == "completed" for t in tasks)
        tests_passed = self._last_test_result is not None and self._last_test_result.passed
        return all_completed and tests_passed

    def _get_tasks(self) -> list[Any]:
        return self._tasks
