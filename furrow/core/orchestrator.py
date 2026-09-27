from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from io import StringIO

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import configure_logging

console = Console()

_RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z][^\]]*\]")

OutputCallback = Callable[[str], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_output: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else None) or LLMClient().settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.current_tasks: list[TaskModel] = []
        self._last_tests_passed: bool = False
        self._no_tasks_planned: bool = False
        self._consecutive_errors: int = 0
        self.on_output: OutputCallback | None = on_output
        self._logger = structlog.get_logger("furrow.orchestrator")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        """Send *text* to the configured output sink.

        When ``on_output`` is set (e.g. web server) Rich renderables are first
        rendered to plain text and markup tags are stripped so the callback
        receives a clean string.  When ``on_output`` is ``None`` Rich
        ``console.print`` is used for formatted CLI output.
        """
        if self.on_output is not None:
            if not isinstance(text, str):
                buf = StringIO()
                plain = Console(
                    file=buf, no_color=True, force_terminal=False, width=100
                )
                plain.print(text)
                text = buf.getvalue().strip()
            text = _RICH_TAG_RE.sub("", text)
            try:
                self.on_output(text)
            except Exception:
                pass
        else:
            console.print(text)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        configure_logging(self.settings.log_level)
        self._logger.info("orchestrator_start", goal=self.goal)

        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            self.cycles += 1
            self._logger.info("cycle_start", cycle=self.cycles, goal=self.goal)

            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                self._emit(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                self._logger.info("max_cycles_reached", cycles=self.cycles)
                break

            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                await self._cycle()
            except Exception as e:
                self._emit(f"[red]Cycle {self.cycles} error: {e}[/red]")
                self._logger.error("cycle_error", cycle=self.cycles, error=str(e))
                self._consecutive_errors += 1
                if self._consecutive_errors >= 3:
                    self._emit("[red]Too many consecutive errors. Stopping.[/red]")
                    break
                self.goal = f"Continue working despite errors. Fix any issues. Original goal: {self.goal}"
            else:
                self._consecutive_errors = 0

            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                self._logger.info("goal_complete", cycles=self.cycles)
                break

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    async def _cycle(self) -> None:
        if self.on_output is None:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.goal)
        else:
            self._emit("Planning...")
            plan = await self.planner.plan(self.goal)

        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.current_tasks = plan.tasks
        self._no_tasks_planned = len(plan.tasks) == 0

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
        else:
            if self.on_output is None:
                with Status("[bold yellow]Executing tasks in parallel...", console=console):
                    results = await asyncio.gather(
                        *[
                            WorkerAgent(task=t, client=self.client).run()
                            for t in plan.tasks
                        ],
                        return_exceptions=True,
                    )
            else:
                self._emit("Executing tasks in parallel...")
                results = await asyncio.gather(
                    *[WorkerAgent(task=t, client=self.client).run() for t in plan.tasks],
                    return_exceptions=True,
                )

            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self._emit(f"[green]Task {task.id} completed[/green]")

        if self.on_output is None:
            with Status("[bold yellow]Testing...", console=console):
                test_result = await self.tester.run(self.goal, plan.tasks)
        else:
            self._emit("Testing...")
            test_result = await self.tester.run(self.goal, plan.tasks)

        self._last_tests_passed = test_result.passed

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  - {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        """Return True when the goal is considered complete.

        The goal is done when:
        - Tests passed **and**
        - No tasks were planned this cycle (the planner has nothing left) **or**
          all planned tasks completed successfully.
        """
        if not self._last_tests_passed:
            return False
        if self._no_tasks_planned:
            return True
        if self.current_tasks and all(
            t.status == "completed" for t in self.current_tasks
        ):
            return True
        # Tasks remain (pending or failed) even though tests passed — keep going.
        return False

    def _get_tasks(self) -> list[TaskModel]:
        """Return the tasks from the most recent plan (for introspection)."""
        return self.current_tasks
