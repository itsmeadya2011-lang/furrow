from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()
logger = logging.getLogger(__name__)


class Orchestrator:
    """Manages the plan-execute-test cycle for achieving a goal.

    The orchestrator repeatedly invokes a planner to generate a plan,
    executes the plan's tasks in parallel via worker agents, and validates
    the results with a tester agent. The loop continues until the goal is
    complete, the cycle limit is reached, or an unrecoverable failure occurs.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            goal: The high-level goal to accomplish.
            client: Optional LLM client to use for all agents.
            max_cycles: Maximum number of cycles before halting. Defaults
                to the configured settings.max_cycles (or 10 if unset).
            settings: Optional Settings instance. Defaults to global settings.
        """
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self._consecutive_empty = 0
        self._consecutive_failures = 0

        resolved_settings = settings or self.client.settings
        default_max = resolved_settings.max_cycles if resolved_settings.max_cycles > 0 else 10
        self.max_cycles = max_cycles if max_cycles is not None else default_max

    async def run(self) -> None:
        """Execute the orchestrator's main loop until completion or halt."""
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self._should_halt_on_failure():
                console.print("[bold red]Too many consecutive failures. Halting.[/bold red]")
                break
            if self._consecutive_empty >= 2:
                console.print("[yellow]Planner returned no tasks twice. Halting.[/yellow]")
                break
            if self.cycles >= self.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.max_cycles}). Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        """Run a single plan-execute-test cycle."""
        plan: Plan | None = None
        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
        except Exception as e:
            logger.error("Planner failed: %s", e)
            console.print(f"[red]Planner failed: {e}[/red]")
            self._consecutive_failures += 1
            self.current_plan = None
            self.last_test_result = None
            return

        self._consecutive_failures = 0
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._consecutive_empty += 1
            self.last_test_result = None
            return

        self._consecutive_empty = 0

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        """Return True if all tasks completed with no failures and tests passed."""
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed != len(tasks):
            return False
        if self.last_test_result is None:
            return False
        return self.last_test_result.passed

    def _should_halt_on_failure(self) -> bool:
        """Return True if too many consecutive planner failures have occurred."""
        return self._consecutive_failures >= 3

    def _get_tasks(self) -> list[TaskModel]:
        """Return the tasks from the most recent plan, or an empty list."""
        if self.current_plan is None:
            return []
        return list(self.current_plan.tasks)
