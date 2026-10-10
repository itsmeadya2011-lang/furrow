from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.text import Text

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the infinite development loop.

    On each cycle the orchestrator asks the planner for tasks, executes
    them in parallel via worker agents (which may read and write files),
    then runs the tester.  Task state is persisted across cycles so the
    loop genuinely iterates until every task is completed or ``max_cycles``
    is reached.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self._max_cycles = self.client.settings.max_cycles
        self._output_callback = output_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self._max_cycles and self.cycles > self._max_cycles:
                self._emit("[yellow]Max cycles reached. Stopping.[/yellow]")
                break
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    async def _cycle(self) -> None:
        # 1. Plan
        context = self._build_context()
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(context)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Register new tasks (unique IDs across cycles)
        self._register_tasks(plan)

        if not plan.tasks:
            self._emit("[yellow]No new tasks planned.[/yellow]")

        # 2. Execute pending tasks in parallel
        pending = [
            t for t in self.all_tasks
            if t.status == "pending"
        ][: self.client.settings.max_parallel_tasks]

        if pending:
            self._emit(f"[cyan]Executing {len(pending)} task(s) in parallel...[/cyan]")
            with Status("[bold yellow]Executing tasks...", console=console):
                coros = [WorkerAgent(task=t, client=self.client).run() for t in pending]
                results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(pending, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self._emit(f"[green]Task {task.id} completed[/green]")
        else:
            self._emit("[dim]No pending tasks to execute.[/dim]")

        # 3. Test
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            # If tests pass, all remaining work is effectively done.
            for t in self.all_tasks:
                if t.status != "completed":
                    t.status = "completed"
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _build_context(self) -> str:
        """Build the goal/context string for the planner, including prior results."""
        parts: list[str] = [self.goal]
        completed = [t for t in self.all_tasks if t.status == "completed"]
        failed = [t for t in self.all_tasks if t.status == "failed"]

        if completed:
            parts.append("\nCompleted tasks in prior cycles (do NOT re-plan these):")
            for t in completed:
                snippet = (t.result or "done")[:300]
                parts.append(f"  - [{t.id}] {t.description}: {snippet}")

        if failed:
            parts.append("\nFailed tasks from prior cycles (address these):")
            for t in failed:
                parts.append(f"  - [{t.id}] {t.description}: {t.result or 'failed'}")

        parts.append(
            "\nBreak the remaining work into 1-5 new tasks. "
            "Do not re-plan already-completed tasks."
        )
        return "\n".join(parts)

    def _register_tasks(self, plan: Plan) -> None:
        """Add tasks from a plan to the tracking list with unique IDs."""
        existing_ids = {t.id for t in self.all_tasks}
        for task in plan.tasks:
            tid = task.id
            if tid in existing_ids:
                tid = f"{self.cycles}-{task.id}"
                task.id = tid
            self.all_tasks.append(task)
            existing_ids.add(tid)

    def _is_done(self) -> bool:
        """Return True when all tracked tasks are completed."""
        if not self.all_tasks:
            return True
        return all(t.status == "completed" for t in self.all_tasks)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _emit(self, msg: Any) -> None:
        """Print to console and optionally forward to an output callback."""
        if self._output_callback:
            try:
                self._output_callback(str(msg))
            except Exception:
                pass
        console.print(msg)
