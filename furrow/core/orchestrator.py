from __future__ import annotations

import asyncio

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


class Orchestrator:
    """Coordinates the infinite parallel development loop.

    Each cycle: plan -> execute tasks in parallel -> test.
    Tasks are accumulated across cycles so completed work is not redone,
    and ``max_cycles`` / ``max_parallel_tasks`` are always enforced.
    """

    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._semaphore: asyncio.Semaphore | None = None
        self._tests_passed: bool = False

    # -- public API ----------------------------------------------------------

    async def run(self) -> None:
        max_cycles = self.client.settings.max_cycles
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                if self._tests_passed:
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                else:
                    console.print(
                        "[yellow]All tasks complete but tests failed. Halting.[/yellow]"
                    )
                break

    # -- cycle internals -----------------------------------------------------

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge newly planned tasks with the existing task list so that
        # previously completed tasks are not re-executed.
        self._merge_tasks(plan.tasks)

        runnable = self._get_runnable_tasks()
        if not runnable:
            completed = sum(1 for t in self.tasks if t.status == "completed")
            console.print(
                f"[yellow]No runnable tasks "
                f"({completed} completed, rest have unmet dependencies).[/yellow]"
            )
        else:
            with Status(
                "[bold yellow]Executing tasks in parallel...", console=console
            ):
                results = await asyncio.gather(
                    *(self._run_task(t) for t in runnable),
                    return_exceptions=True,
                )

            for task, result in zip(runnable, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            self._tests_passed = True
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._tests_passed = False
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    # -- task management -----------------------------------------------------

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        """Integrate newly planned tasks with the existing task list.

        Existing completed tasks keep their status.  Existing pending or
        failed tasks are updated with the latest description / dependencies.
        Genuinely new tasks are appended.
        """
        existing_by_id: dict[str, TaskModel] = {t.id: t for t in self.tasks}
        for task in new_tasks:
            if task.id in existing_by_id:
                existing = existing_by_id[task.id]
                if existing.status != "completed":
                    # Reset failed/running tasks back to pending so they
                    # can be retried in a subsequent cycle.
                    existing.status = "pending"
                    existing.description = task.description
                    if task.files:
                        existing.files = task.files
                    if task.dependencies:
                        existing.dependencies = task.dependencies
            else:
                self.tasks.append(task)

    def _get_runnable_tasks(self) -> list[TaskModel]:
        """Return pending tasks whose dependencies are all completed."""
        completed_ids = {t.id for t in self.tasks if t.status == "completed"}
        runnable = [
            t
            for t in self.tasks
            if t.status == "pending"
            and all(dep in completed_ids for dep in t.dependencies)
        ]
        return runnable

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            limit = max(1, self.client.settings.max_parallel_tasks)
            self._semaphore = asyncio.Semaphore(limit)
        return self._semaphore

    async def _run_task(self, task: TaskModel) -> str:
        async with self.semaphore:
            task.status = "running"
            worker = WorkerAgent(task=task, client=self.client)
            return await worker.run()

    # -- termination checks --------------------------------------------------

    def _is_done(self) -> bool:
        """Return ``True`` when no further useful work can be done.

        The loop terminates when every task is in a terminal state
        (``completed`` / ``failed`` / ``skipped``).  Failed tasks cause the
        loop to continue — a subsequent cycle will re-plan fix-up tasks.
        """
        if not self.tasks:
            return True
        terminal = {"completed", "failed", "skipped"}
        return all(t.status in terminal for t in self.tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
