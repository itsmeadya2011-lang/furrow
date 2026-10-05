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
from furrow.config import Settings, TaskModel, settings as default_settings
from furrow.llm import LLMClient

console = Console()
_plain_console = Console(color_system=None, width=console.width)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_output: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._tests_passed: bool = False
        self._on_output = on_output

    async def run(self, on_output=None) -> None:
        if on_output is not None:
            self._on_output = on_output
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _emit(self, text: Any = "") -> None:
        console.print(text)
        if self._on_output is not None:
            with _plain_console.capture() as cap:
                _plain_console.print(text)
            await self._on_output(cap.get())

    def _topological_sort(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Order tasks so that dependencies come before the tasks that depend on them."""
        by_id = {t.id: t for t in tasks}
        ordered: list[TaskModel] = []
        visited: set[str] = set()
        temp: set[str] = set()

        def visit(task: TaskModel) -> None:
            if task.id in visited:
                return
            if task.id in temp:
                # Cycle detected — break it by ignoring the dependency
                return
            temp.add(task.id)
            for dep_id in task.dependencies:
                if dep_id in by_id:
                    visit(by_id[dep_id])
            temp.discard(task.id)
            visited.add(task.id)
            ordered.append(task)

        for task in tasks:
            visit(task)
        return ordered

    async def _execute_task_with_retries(self, task: TaskModel) -> str:
        """Execute a single task, retrying up to 2 times on failure."""
        max_retries = 2
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                return await worker.run()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    task.status = "retrying"
                    task.result = f"Attempt {attempt + 1} failed: {e}"
                    await self._emit(f"[yellow]Task {task.id} retry {attempt + 1}/{max_retries}: {e}[/yellow]")
                else:
                    raise
        # Should not reach here, but guard against it
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Task {task.id} failed after retries")

    async def _execute_wave(self, tasks: list[TaskModel]) -> None:
        """Execute a wave of independent tasks in parallel."""
        coros = [self._execute_task_with_retries(task) for task in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

    async def _cycle(self) -> None:
        await self._emit("Planning...")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store tasks for tracking across cycles
        self._tasks = plan.tasks

        # Order tasks topologically so dependencies run first
        ordered_tasks = self._topological_sort(plan.tasks)

        # Group into dependency waves: tasks that can run in parallel
        completed_ids: set[str] = set()
        waves: list[list[TaskModel]] = []
        remaining = list(ordered_tasks)
        while remaining:
            wave = [t for t in remaining if all(d in completed_ids for d in t.dependencies)]
            if not wave:
                # Safety: if no progress can be made (e.g. cycle), run the rest
                wave = remaining
            waves.append(wave)
            for t in wave:
                completed_ids.add(t.id)
            remaining = [t for t in remaining if t not in wave]

        await self._emit("Executing tasks in waves...")
        with Status("[bold yellow]Executing tasks in waves...", console=console):
            for wave in waves:
                await self._execute_wave(wave)

        await self._emit("Testing...")
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self._tasks)

        self._tests_passed = test_result.passed

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        # Enforce max_cycles limit
        if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
            return True

        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")

        # If any task failed, not done — keep going (goal updated to fix failures)
        if failed > 0:
            return False

        # All tasks completed
        if self._tasks and completed >= len(self._tasks):
            # Only stop when tests also passed
            return self._tests_passed

        return False

    def _get_tasks(self) -> list[Any]:
        return self._tasks