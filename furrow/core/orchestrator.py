from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)

OutputSink = Callable[[str], None]


def _strip_markup(message: str) -> str:
    """Extract plain text from a Rich markup string."""
    return Text.from_markup(message).plain


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_sink: OutputSink | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self._mid_loop_input: str | None = None
        self.output_sink = output_sink

    def add_input(self, text: str) -> None:
        """Incorporate mid-loop user input into the next plan."""
        self._mid_loop_input = text

    def _broadcast(self, message: str) -> None:
        """Send a message to the console and optionally to the output sink.

        Accepts Rich markup strings (e.g. '[bold green]Done[/bold green]').
        For rendering Rich objects (Panel, etc.), use _broadcast_panel instead.
        """
        console.print(message)
        if self.output_sink:
            self.output_sink(_strip_markup(message))

    def _broadcast_panel(self, panel: Any, plain_text: str) -> None:
        """Render a Rich Panel to console and send plain text to the sink."""
        console.print(panel)
        if self.output_sink:
            self.output_sink(plain_text)

    async def run(self) -> None:
        self._broadcast_panel(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"),
            f"Furrow - Goal: {self.goal}",
        )
        logger.info("orchestrator_start", goal=self.goal, max_cycles=self.settings.max_cycles)

        while True:
            # Enforce max_cycles (0 means unlimited)
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._broadcast(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles)
                break

            self.cycles += 1
            self._broadcast(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            # Incorporate any mid-loop user input
            current_goal = self.goal
            if self._mid_loop_input:
                current_goal = self._mid_loop_input
                self._broadcast(f"[cyan]Mid-loop input:[/cyan] {self._mid_loop_input}")
                self._mid_loop_input = None

            await self._cycle(current_goal)

            if self._is_done():
                self._broadcast("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self, goal: str) -> None:
        plan = await self.planner.plan(goal)
        self.current_plan = plan
        plan_text = f"Plan ({plan.rationale}): {', '.join(t.id for t in plan.tasks)}"
        self._broadcast_panel(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"),
            plan_text,
        )

        if not plan.tasks:
            self._broadcast("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._broadcast("[yellow]Executing tasks in parallel...[/yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._broadcast(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self._broadcast(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        self._broadcast("[yellow]Testing...[/yellow]")
        test_result = await self.tester.run(goal, plan.tasks)

        if test_result.passed:
            self._broadcast(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            self._broadcast(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._broadcast(f"  • {failure}")
            self._broadcast("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        """Check whether the goal is complete based on the latest plan."""
        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            # Failed tasks will be retried next cycle
            return False

        # Goal is done when all tasks in the plan are completed
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        """Return the tasks from the current plan, or empty list if no plan exists."""
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
