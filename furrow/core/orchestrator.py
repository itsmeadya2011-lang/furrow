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
from furrow.config import Plan, get_logger
from furrow.llm import LLMClient

# Module-level default console (used when none is injected, e.g. CLI usage).
default_console = Console()
log = get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self.console: Console = console if console is not None else default_console

    async def run(self) -> None:
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            if max_cycles > 0 and self.cycles > max_cycles:
                self.console.print(
                    f"[yellow]Reached max_cycles ({max_cycles}). Halting.[/yellow]"
                )
                log.info("orchestrator.max_cycles_reached", cycles=self.cycles)
                break
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("orchestrator.goal_complete", cycles=self.cycles)
                break

    @property
    def _sem(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            limit = self.client.settings.max_parallel_tasks
            self._semaphore = asyncio.Semaphore(limit)
        return self._semaphore

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            coros = [self._run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("worker.failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                log.info("worker.completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("test.passed", summary=test_result.summary)
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            log.warning(
                "test.failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, task: Any) -> str:
        """Run a single worker under the parallelism semaphore."""
        async with self._sem:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        """Return tasks from the most recent plan, or [] if no plan exists."""
        if self.plan is not None:
            return self.plan.tasks
        return []
