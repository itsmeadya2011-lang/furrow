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
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        log_func: Any = None,
        rich_console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else Settings())
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.rich_console = rich_console or console
        self.cycles = 0
        self.tasks: list[Any] = []
        self.last_test_result: TestResult | None = None
        self.log_func = log_func
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.settings.max_parallel_tasks
        )

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print to the configured console (default or custom for web)."""
        self.rich_console.print(*args, **kwargs)

    def _log(self, msg: str) -> None:
        self.rich_console.print(f"[dim]Orchestrator:[/dim] {msg}")
        if self.log_func:
            self.log_func(msg)

    async def run(self) -> None:
        self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._log(f"Starting Furrow with goal: {self.goal}")

        while True:
            # Enforce max_cycles
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._log("Max cycles reached. Stopping.")
                self._print("[yellow]Max cycles reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._log(f"Starting cycle {self.cycles}")

            await self._cycle()

            if self._is_done():
                if self.last_test_result and self.last_test_result.passed:
                    self._print("[bold green]Goal complete. Halting.[/bold green]")
                    self._log("Goal complete — all tests passed.")
                else:
                    self._print("[yellow]No more tasks to plan. Halting.[/yellow]")
                    self._log("Halting — no more tasks.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.rich_console):
            plan = await self.planner.plan(self.goal)
        self._print(Panel(Pretty(self._plan_summary(plan)), title="Plan", border_style="blue"))
        self._log(f"Planned {len(plan.tasks)} task(s).")

        if not plan.tasks:
            self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._log("No tasks planned.")
            return

        # Store tasks for _is_done() tracking
        self.tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=self.rich_console):
            coros = [self._run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._print(f"[red]Task {task.id} failed: {result}[/red]")
                self._log(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                self._print(f"[green]Task {task.id} completed[/green]")
                self._log(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=self.rich_console):
            test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, plan.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._log("Tests passed.")
        else:
            self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._print(f"  • {failure}")
            self._log(f"Tests failed: {test_result.summary}")
            self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, task: Any) -> str:
        """Run a single worker task with parallelism limiting via semaphore."""
        async with self._semaphore:
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            return await worker.run()

    def _plan_summary(self, plan: Plan) -> dict[str, Any]:
        return {
            "tasks": [
                {"id": t.id, "description": t.description, "files": t.files}
                for t in plan.tasks
            ],
            "rationale": plan.rationale,
        }

    def _is_done(self) -> bool:
        # No tasks — nothing left to do
        if not self.tasks:
            return True

        # If tests have run and passed, we're done
        if self.last_test_result is not None:
            if self.last_test_result.passed:
                return True

        # Otherwise keep going (will re-plan to fix failures)
        return False
