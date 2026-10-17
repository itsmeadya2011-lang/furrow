from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            if settings.max_cycles > 0 and self.cycles > settings.max_cycles:
                console.print(
                    f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]"
                )
                break

            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks...", console=console):
            results = await self._execute_tasks(plan.tasks)

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

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[Any]) -> list[Any]:
        """Execute tasks, honoring dependencies: dependent tasks run sequentially
        after their dependencies complete; independent tasks run in parallel."""
        pending = list(tasks)
        completed_ids: set[str] = set()
        results_map: dict[str, Any] = {}

        async def run_one(task: Any) -> Any:
            worker = WorkerAgent(task=task, client=self.client)
            return await worker.run()

        while pending:
            ready = [
                t
                for t in pending
                if all(dep in completed_ids for dep in (t.dependencies or []))
            ]
            if not ready:
                # Circular or unresolvable dependencies — run remaining in parallel
                ready = pending

            batch = [
                asyncio.create_task(run_one(t)) for t in ready
            ]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for task, result in zip(ready, batch_results):
                results_map[task.id] = result
                completed_ids.add(task.id)
                pending.remove(task)

        return [results_map[t.id] for t in tasks]

    def _is_done(self) -> bool:
        if not self.tasks:
            return True  # nothing planned = nothing left to do
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks
