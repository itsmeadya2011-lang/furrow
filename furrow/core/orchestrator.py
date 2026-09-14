from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console
from rich.markup import strip as strip_markup
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Callable[[str], None] | Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._all_tasks: list[Any] = []
        self._on_output = on_output

    def _emit(self, message: str) -> None:
        """Send a message to the console and any registered output callback."""
        console.print(message)
        if self._on_output is not None:
            plain = strip_markup(message)
            result = self._on_output(plain)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._emit(f"[bold yellow]Reached max cycles ({max_cycles}). Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks into the tracked task list, avoiding duplicates by id.
        existing_ids = {t.id for t in self._all_tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self._all_tasks.append(task)
                existing_ids.add(task.id)

        # Resolve dependencies: only run tasks whose dependencies are satisfied.
        ready = self._ready_tasks()
        if not ready:
            self._emit("[yellow]No tasks ready to run (dependencies unmet).[/yellow]")
            return

        max_parallel = self.client.settings.max_parallel_tasks
        # Execute in bounded batches to respect max_parallel_tasks.
        for batch_start in range(0, len(ready), max_parallel):
            batch = ready[batch_start : batch_start + max_parallel]
            with Status(
                f"[bold yellow]Executing {len(batch)} task(s) in parallel...[/bold yellow]",
                console=console,
            ):
                coros = [WorkerAgent(task=task, client=self.client).run() for task in batch]
                results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self._all_tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _ready_tasks(self) -> list[Any]:
        """Return pending tasks whose dependencies are all satisfied."""
        completed_ids = {t.id for t in self._all_tasks if t.status == "completed"}
        return [
            t
            for t in self._all_tasks
            if t.status == "pending" and all(dep in completed_ids for dep in t.dependencies)
        ]

    def _is_done(self) -> bool:
        tasks = self._all_tasks
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        # If any task failed, we're not done (the fix cycle will retry).
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self._all_tasks
