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
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Infinite-loop orchestrator that plans, executes, and tests in parallel.

    The orchestrator keeps the original goal across cycles and only re-plans
    when the planner needs to retry.  It honors ``Settings.max_parallel_tasks``
    and ``Settings.max_cycles`` (``0`` means infinite).
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        from furrow.config import settings as default_settings

        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test: TestResult | None = None
        self._stopped = False

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while not self._stopped:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Cycle failed: {exc}[/red]")
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if (
                self.settings.max_cycles > 0
                and self.cycles >= self.settings.max_cycles
            ):
                console.print(
                    f"[yellow]Reached max_cycles={self.settings.max_cycles}. Halting.[/yellow]"
                )
                break

    def stop(self) -> None:
        """Request the loop to exit at the next cycle boundary."""
        self._stopped = True

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._tasks = plan.tasks
        ready = self._ready_tasks(plan)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

            async def _run(task: TaskModel) -> tuple[TaskModel, Any]:
                async with sem:
                    return task, await WorkerAgent(
                        task=task, client=self.client, settings=self.settings
                    ).run()

            results = await asyncio.gather(
                *(_run(t) for t in ready), return_exceptions=True
            )

        for item in results:
            if isinstance(item, Exception):
                console.print(f"[red]Task dispatch failed: {item}[/red]")
                continue
            task, result = item
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        # Mark any non-ready tasks as still pending (waiting on deps).
        ready_ids = {t.id for t in ready}
        for t in plan.tasks:
            if t.id not in ready_ids and t.status == "pending":
                console.print(
                    f"[dim]Task {t.id} deferred (waiting on dependencies).[/dim]"
                )

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, plan.tasks)
        self._last_test = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Preserve original goal context for the next plan.
            self.goal = (
                f"{self.goal}\n\nFix the following test failures:\n"
                + "\n".join(test_result.failures)
            )

    def _ready_tasks(self, plan: Plan) -> list[TaskModel]:
        """Return tasks whose declared dependencies are all completed."""
        completed_ids = {
            t.id for t in self._tasks if t.status == "completed"
        } | {
            t.id for t in plan.tasks if t.status == "completed"
        }
        ready: list[TaskModel] = []
        for t in plan.tasks:
            if t.status == "completed":
                continue
            if all(dep in completed_ids for dep in t.dependencies):
                ready.append(t)
        return ready

    def _is_done(self) -> bool:
        if not self._tasks:
            return False
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        pending = sum(1 for t in self._tasks if t.status == "pending")
        if failed > 0:
            return False
        if pending > 0:
            return False
        return completed == len(self._tasks)

    @property
    def tasks(self) -> list[TaskModel]:
        """Public read-only view of the tasks from the latest plan."""
        return list(self._tasks)

    @property
    def last_test(self) -> TestResult | None:
        """The most recent test result, or ``None`` if no tests have run yet."""
        return self._last_test
