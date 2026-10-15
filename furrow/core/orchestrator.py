from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
_logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        *,
        output_callback: Any = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        # Track all tasks across cycles so _get_tasks works correctly
        self.tasks: dict[str, TaskModel] = {}
        # Optional callback for live output (used by web server)
        self._output_callback = output_callback
        # Per-task retry counter and limit
        self._retry_limit = 2
        self._last_test_passed = True

    # -- output helpers --------------------------------------------------

    def _emit(self, message: str) -> None:
        """Send output to console and/or callback.

        The callback receives plain-text (Rich markup stripped) so it can be
        forwarded to non-terminal consumers such as WebSocket clients.
        """
        console.print(message)
        if self._output_callback:
            from rich.console import Console

            plain_console = Console(force_terminal=False, no_color=True, emoji=False)
            with plain_console.capture() as cap:
                plain_console.print(message)
            text = cap.get()
            self._output_callback(text)

    # -- main loop -------------------------------------------------------

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1

            # Enforce max_cycles (0 = unlimited)
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                self._emit("[yellow]Max cycles reached. Stopping.[/yellow]")
                _logger.info("Max cycles reached, stopping", cycles=self.cycles)
                break

            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            _logger.info("Starting cycle", cycle=self.cycles, goal=self.goal)
            await self._cycle()

            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                _logger.info("Goal complete, halting", cycle=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)

        # Merge newly planned tasks into the tracked set
        for task in plan.tasks:
            self.tasks[task.id] = task

        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            _logger.info("No tasks in plan")
            return

        # Execute tasks respecting dependencies and parallelism limit
        await self._execute_tasks(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(
                self.goal, list(self.tasks.values())
            )

        if test_result.passed:
            self._last_test_passed = True
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._last_test_passed = False
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            _logger.info("Tests failed, will retry", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Execute tasks in dependency order with a parallelism limit."""
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_limit(task: TaskModel) -> None:
            async with semaphore:
                # Retry logic for failed tasks
                attempt = getattr(task, "_attempts", 0)
                while attempt < self._retry_limit:
                    try:
                        result = await WorkerAgent(task=task, client=self.client).run()
                        task.status = "completed"
                        task.result = result
                        self._emit(f"[green]Task {task.id} completed[/green]")
                        _logger.info("Task completed", task_id=task.id)
                        return
                    except Exception as e:
                        attempt += 1
                        task._attempts = attempt
                        _logger.warning(
                            "Task attempt failed, retrying",
                            task_id=task.id,
                            attempt=attempt,
                            error=str(e),
                        )
                        if attempt >= self._retry_limit:
                            task.status = "failed"
                            task.result = str(e)
                            self._emit(f"[red]Task {task.id} failed after {attempt} attempts: {e}[/red]")
                            _logger.error("Task failed after retries", task_id=task.id, attempts=attempt)
                            return
                        await asyncio.sleep(0.5 * attempt)  # Brief backoff before retry

        # Build dependency-aware execution order
        # Tasks with no unmet dependencies run first; we process in waves
        pending_ids = {t.id for t in tasks if t.status == "pending"}
        completed_ids: set[str] = set()

        while pending_ids:
            # Find tasks whose dependencies are all satisfied
            ready = [
                t for t in tasks
                if t.id in pending_ids
                and all(dep in completed_ids or dep not in {x.id for x in tasks} for dep in t.dependencies)
            ]

            if not ready:
                # Circular dependency or missing dependency — execute remaining tasks
                ready = [t for t in tasks if t.id in pending_ids]

            current_wave = ready[: self.settings.max_parallel_tasks]
            pending_ids -= {t.id for t in current_wave}

            await asyncio.gather(
                *(run_with_limit(task) for task in current_wave),
                return_exceptions=True,
            )

            completed_ids.update({t.id for t in current_wave if t.status == "completed"})
            pending_ids -= completed_ids

    def _is_done(self) -> bool:
        all_tasks = self._get_tasks()
        completed = sum(1 for t in all_tasks if t.status == "completed")
        failed = sum(1 for t in all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if not all_tasks:
            return True
        # Done only if all tasks completed AND tests passed
        if completed >= len(all_tasks):
            return self._last_test_passed
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return list(self.tasks.values())
