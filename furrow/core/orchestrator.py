from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient

console = Console()

PLAN_ATTEMPTS = 3


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.focus = goal
        self.client = client or LLMClient()
        self.settings: Settings = client.settings if client is not None else settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self._semaphore: asyncio.Semaphore | None = None

    def _max_cycles_reached(self) -> bool:
        return self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles

    def _layer_plan(self, tasks: list[TaskModel]) -> tuple[list[list[TaskModel]], list[str]]:
        """Core layering pass shared by ``_dependency_layers`` and cycle detection.

        Returns the dependency layers plus the ids of the tasks that could only be
        flushed by the fallback layer because none of them was runnable any more.
        Tasks are tracked by their original index so duplicate ids stay distinct.
        """
        known = {task.id for task in tasks}
        pending = list(enumerate(tasks))
        satisfied: set[str] = set()
        layers: list[list[TaskModel]] = []
        cycle_ids: list[str] = []

        while pending:
            runnable = [
                (index, task)
                for index, task in pending
                if all(dep not in known or dep in satisfied for dep in task.dependencies)
            ]
            if not runnable:
                # A dependency cycle (or a self-dependency) is left. Flush the
                # remainder as one final layer instead of spinning forever.
                cycle_ids = [task.id for _, task in pending]
                runnable = pending
            taken = {index for index, _ in runnable}
            layers.append([task for _, task in runnable])
            satisfied.update(task.id for _, task in runnable)
            pending = [(index, task) for index, task in pending if index not in taken]

        return layers, cycle_ids

    def _dependency_layers(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group tasks into dependency layers.

        Tasks within a layer may run concurrently; each layer runs only after the
        previous one finishes. Dependencies naming a task that is not in ``tasks``
        are treated as already satisfied. If the dependency graph contains a cycle,
        the remaining tasks are emitted in one final layer so the loop cannot
        deadlock.
        """
        layers, _ = self._layer_plan(tasks)
        return layers

    def _dependency_cycle_ids(self, tasks: list[TaskModel]) -> list[str]:
        """Ids of tasks stuck in a dependency cycle; empty when the graph is clean."""
        _, cycle_ids = self._layer_plan(tasks)
        return cycle_ids

    async def run(self) -> None:
        banner = f"[bold green]Furrow[/bold green]\nGoal: {self.goal}"
        console.print(Panel.fit(banner, title="Furrow"))
        while True:
            if self._max_cycles_reached():
                console.print(
                    f"[yellow]Cycle limit reached ({self.settings.max_cycles}). Halting.[/yellow]"
                )
                break
            self.cycles += 1
            limit = f"/{self.settings.max_cycles}" if self.settings.max_cycles > 0 else ""
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles}{limit} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Plan against the real goal unless the previous cycle redirected us.
        self.focus = self.goal
        last_error: ValueError | None = None
        for attempt in range(1, PLAN_ATTEMPTS + 1):
            try:
                with Status(f"[bold yellow]Planning (attempt {attempt})...", console=console):
                    plan = await self.planner.plan(self.focus)
            except ValueError as exc:
                last_error = exc
                console.print(
                    f"[yellow]Planning attempt {attempt}/{PLAN_ATTEMPTS} failed: {exc}[/yellow]"
                )
                continue
            break
        else:
            message = f"Planner failed after {PLAN_ATTEMPTS} attempts: {last_error}"
            console.print(f"[red]{message}[/red]")
            raise RuntimeError(message) from last_error

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.all_tasks.extend(plan.tasks)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

        cycle_ids = self._dependency_cycle_ids(plan.tasks)
        if cycle_ids:
            names = ", ".join(cycle_ids)
            console.print(
                f"[yellow]Dependency cycle detected between tasks: {names}. "
                "Running them concurrently.[/yellow]"
            )

        for layer in self._dependency_layers(plan.tasks):
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                running = [self._run_task(task) for task in layer]
                results = await asyncio.gather(*running, return_exceptions=True)

            for task, result in zip(layer, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = f"{type(result).__name__}: {result}"
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.focus = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_task(self, task: TaskModel) -> str:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        return bool(self.all_tasks) and all(t.status == "completed" for t in self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
