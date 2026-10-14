from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[Any] = []

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]"
                )
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            has_tasks = await self._cycle()
            if not has_tasks:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> bool:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._tasks = []
            return False

        self._tasks = plan.tasks

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=console
        ):
            await self._execute_tasks(plan.tasks)

        for task in self._tasks:
            if task.status == "completed":
                console.print(f"[green]Task {task.id} completed[/green]")
            elif task.status == "failed":
                console.print(f"[red]Task {task.id} failed: {task.result}[/red]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, self._tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return True

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        """Execute tasks respecting dependencies and parallelism limits.

        Tasks are grouped into batches: every task whose dependencies have all
        completed is eligible to run. Within a batch, tasks run concurrently up
        to ``max_parallel_tasks``.
        """
        completed_ids: set[str] = set()
        max_parallel = self.client.settings.max_parallel_tasks or 5
        semaphore = asyncio.Semaphore(max_parallel)

        remaining = list(tasks)
        while remaining:
            ready = [
                t for t in remaining if all(dep in completed_ids for dep in t.dependencies)
            ]
            if not ready:
                # No task is ready: either a missing/circular dependency.
                pending_ids = {t.id for t in remaining}
                console.print(
                    f"[red]No runnable tasks among {pending_ids}. "
                    "Possible circular or missing dependency.[/red]"
                )
                for t in remaining:
                    t.status = "failed"
                    t.result = "Unresolvable dependency"
                break

            async def _run_one(task: Any) -> Any:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            results = await asyncio.gather(
                *(_run_one(t) for t in ready), return_exceptions=True
            )

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                else:
                    task.status = "completed"
                    task.result = result
                completed_ids.add(task.id)

            remaining = [t for t in remaining if t.id not in completed_ids]

    def _is_done(self) -> bool:
        if not self._tasks:
            return True
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        # Done when every task finished and none failed.
        return failed == 0 and completed >= len(self._tasks)

    def _get_tasks(self) -> list[Any]:
        return self._tasks
