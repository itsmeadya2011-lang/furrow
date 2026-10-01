from __future__ import annotations

import asyncio

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
    """Runs the plan -> execute -> test loop until tests pass or max_cycles."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        # `console` parameter shadows the module global, so fall back to a fresh
        # Console() when none is injected (e.g. CLI usage).
        self.console = console if console is not None else Console()
        self.settings = self.client.settings
        self.tasks: list[TaskModel] = []
        self.cycles = 0

    async def run(self) -> None:
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            test_result = await self._cycle()
            if test_result.passed:
                self.console.print(
                    "[bold green]Tests passed. Goal complete. Halting.[/bold green]"
                )
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.console.print("[bold yellow]Reached max_cycles. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> TestResult:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")

        sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

        async def _run(task: TaskModel) -> TaskModel:
            try:
                async with sem:
                    result = await WorkerAgent(task=task, client=self.client).run()
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
            except Exception as e:  # noqa: BLE001
                task.status = "failed"
                task.result = str(e)
                self.console.print(f"[red]Task {task.id} failed: {e}[/red]")
            return task

        await asyncio.gather(
            *[_run(task) for task in plan.tasks], return_exceptions=True
        )

        # Persist tasks across cycles: update in place by id, else append.
        existing = {t.id: i for i, t in enumerate(self.tasks)}
        for task in plan.tasks:
            if task.id in existing:
                self.tasks[existing[task.id]] = task
            else:
                self.tasks.append(task)

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return test_result

    def _get_tasks(self) -> list[TaskModel]:
        """Return the accumulated task history across cycles."""
        return self.tasks
