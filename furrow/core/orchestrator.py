from __future__ import annotations

import asyncio

from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, log=None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._log_sink = log
        self.last_test_passed: bool | None = None

    async def _emit(self, msg: str) -> None:
        if self._log_sink is not None:
            await self._log_sink(msg)
        else:
            console.print(msg)

    async def run(self) -> None:
        await self._emit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                failed = sum(1 for t in self.tasks if t.status == "failed")
                if failed:
                    await self._emit(
                        f"[bold yellow]{failed} task(s) could not be completed after retries. Halting.[/bold yellow]"
                    )
                else:
                    await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                await self._emit("[bold yellow]Reached max_cycles limit. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        plan = await self.planner.plan(self.goal)

        existing = {t.id for t in self.tasks}
        for t in plan.tasks:
            if t.id not in existing:
                self.tasks.append(t)
                existing.add(t.id)

        if not self.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        completed_ids = {t.id for t in self.tasks if t.status == "completed"}
        runnable = [
            t for t in self.tasks
            if t.status in ("pending", "failed")
            and t.retries < settings.max_retries
            and all(dep in completed_ids for dep in t.dependencies)
        ]

        if not runnable:
            runnable = [
                t for t in self.tasks
                if t.status in ("pending", "failed")
                and t.retries < settings.max_retries
            ]

        runnable = runnable[: settings.max_parallel_tasks]

        if not runnable:
            await self._emit("[yellow]No runnable tasks this cycle.[/yellow]")
            return

        await self._emit(f"Executing {len(runnable)} task(s) in parallel...")

        workers = [WorkerAgent(task=t, client=self.client).run() for t in runnable]
        results = await asyncio.gather(*workers, return_exceptions=True)

        for t, r in zip(runnable, results):
            if isinstance(r, Exception):
                t.status = "failed"
                t.retries += 1
                t.result = str(r)
                await self._emit(f"[red]Task {t.id} failed: {r}[/red]")
            else:
                t.status = "completed"
                t.result = r
                await self._emit(f"[green]Task {t.id} completed[/green]")

        test_result = await self.tester.run(self.goal, self.tasks)
        self.last_test_passed = test_result.passed

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        # If the last test run failed, keep looping so the (mutated) fix goal is attempted.
        if self.last_test_passed is False:
            return False
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if completed + failed >= len(self.tasks):
            return True
        return False
