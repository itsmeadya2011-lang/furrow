from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

# Module-level default console
_console = Console()


class Orchestrator:
    """Coordinates the infinite parallel development loop: plan, execute, test."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else Settings())
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self._console = console or _console
        self._on_output = on_output
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []

    def _output(self, *args: object, **kwargs: object) -> None:
        """Print to console and optionally relay to an output handler."""
        self._console.print(*args, **kwargs)
        if self._on_output is not None:
            # Extract plain text from rich renderables
            with self._console.capture() as capture:
                self._console.print(*args, **kwargs)
            text = capture.get_text()
            self._on_output(text)

    async def run(self) -> None:
        logger.info("Starting orchestrator for goal: %s", self.goal)
        self._output(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._output(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                self._output(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]")
                logger.info("Reached max_cycles, halting")
                break

            await self._cycle()

            if self._is_done():
                self._output("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("Goal complete after %d cycle(s)", self.cycles)
                break

    async def _cycle(self) -> None:
        plan = await self.planner.plan(self.goal)
        self._output(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._output("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks respecting dependencies and concurrency limits
        results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._output(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._output(f"[green]Task {task.id} completed[/green]")

        # Track tasks for cross-cycle completion checking
        self.all_tasks.extend(plan.tasks)

        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._output(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._output(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._output(f"  • {failure}")
            self._output("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[str | Exception]:
        """Execute tasks in parallel, respecting dependencies and max_parallel_tasks."""
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        done: dict[str, str | Exception] = {}

        async def _run_task(task: TaskModel) -> str | Exception:
            async with semaphore:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    done[task.id] = result
                    return result
                except Exception as e:  # noqa: BLE001 — intentional: catch all worker errors
                    logger.error("Worker task %s failed: %s", task.id, e)
                    done[task.id] = e
                    return e

        # Topological grouping: run tasks with satisfied dependencies first,
        # then their dependents. If no progress can be made (circular dep),
        # run all remaining tasks at once.
        pending = {t.id: t for t in tasks}

        while pending:
            ready = [
                t for t in pending.values()
                if all(dep in done for dep in t.dependencies)
            ]
            if not ready:
                # Circular or missing dependency — run remaining in parallel
                ready = list(pending.values())

            coros = [_run_task(t) for t in ready]
            await asyncio.gather(*coros, return_exceptions=True)

            for t in ready:
                del pending[t.id]

        # Return results in the original task order
        return [done.get(t.id, Exception("Task did not run")) for t in tasks]

    def _is_done(self) -> bool:
        """Check if all tasks across all cycles completed successfully."""
        if not self.all_tasks:
            return True
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        return failed == 0 and completed >= len(self.all_tasks)
