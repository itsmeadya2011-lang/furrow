from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the infinite development loop for a given goal."""

    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self._goal_complete = False

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._should_stop():
                if self._goal_complete:
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                else:
                    console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._goal_complete = True
            return

        # Only execute tasks that haven't already been completed in a prior cycle.
        pending_tasks = [t for t in plan.tasks if t.status != "completed"]
        if not pending_tasks:
            self._goal_complete = True
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, pending_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            # If every task is now completed and tests pass, the goal is done.
            if all(t.status == "completed" for t in plan.tasks):
                self._goal_complete = True
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  [red]\u2022[/red] {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Refine the goal for the next cycle so the planner focuses on fixes.
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _should_stop(self) -> bool:
        """Return True when the loop should halt."""
        # Respect the max_cycles setting (0 means unlimited).
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        return bool(self._goal_complete)

    def _get_tasks(self) -> list[TaskModel]:
        """Return the tasks from the most recent plan (used for introspection)."""
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
