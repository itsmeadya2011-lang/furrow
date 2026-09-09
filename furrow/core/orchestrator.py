from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger("furrow.orchestrator")
console = Console()


class Orchestrator:
    """Orchestrates the planning, execution, and testing loop for a goal.

    The orchestrator repeatedly plans tasks, executes them in parallel via
    worker agents, and validates the outcome with a tester agent.  It halts
    when all tasks have completed successfully with no test failures, or when
    a maximum number of cycles is reached.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            goal: The high-level goal the orchestrator should achieve.
            client: Optional pre-configured LLM client.  A new one is created
                when this is omitted.
            settings: Optional settings instance used to create the default
                LLM client when ``client`` is not provided.
            console: Optional Rich Console instance for output.  The module-level
                console is used when this is omitted.
        """
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()
        self.goal = goal
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._console = console or globals()["console"]

    async def run(self) -> None:
        """Run the orchestrator loop until the goal is complete or halted.

        Each iteration increments the cycle counter, runs a single
        planning/execution/testing cycle, and then checks whether the goal is
        done.  When done, a completion message is printed and the loop exits.
        """
        logger.info("orchestrator_started", goal=self.goal)
        self._console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            self._console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_complete", cycles=self.cycles)
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self._console.print(f"[yellow]Reached max cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
                logger.info("orchestrator_max_cycles_reached", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        """Run a single planning, execution, and testing cycle.

        The cycle:
            1. Plans tasks for the current goal.
            2. Executes the tasks in parallel using worker agents.
            3. Tests the outcome with a tester agent.

        If tests fail, the goal is updated to describe the failing tests so the
        next cycle attempts to fix them.
        """
        with Status("[bold yellow]Planning...", console=self._console) as status:
            plan = await self.planner.plan(self.goal)
        self._console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._current_plan = plan

        if not plan.tasks:
            self._console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self._console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self._console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._console.print(f"  • {failure}")
            self._console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        """Determine whether the orchestrator has completed its goal.

        The orchestrator is considered done when all tasks from the current
        plan have completed successfully and there are no test failures.  An
        empty task list (no plan or no tasks) is also treated as done, unless
        tests have failed, in which case more cycles are needed.

        Returns:
            True if the goal is complete, False otherwise.
        """
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            return False
        if len(tasks) == 0:
            return True
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        """Return the tasks from the current plan.

        Returns:
            The list of tasks in ``self._current_plan``, or an empty list when
            no plan has been set yet.
        """
        if self._current_plan is None:
            return []
        return self._current_plan.tasks