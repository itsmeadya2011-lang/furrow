from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.settings = settings
        if self.settings is None:
            self.settings = getattr(client, "settings", None)
        if self.settings is None:
            self.settings = Settings()
        if client is None:
            client = LLMClient(settings=self.settings)
        self.client = client
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.latest_test_result: TestResult | None = None
        self.completed = False
        self.halt_reason: str | None = None
        self.cycles = 0
        self.max_cycles = self.settings.max_cycles

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while not self.completed and not self.halt_reason:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self.halt_reason = f"Maximum cycle limit of {self.max_cycles} reached."
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

        if self.completed:
            console.print("[bold green]Goal complete. Halting.[/bold green]")
        else:
            reason = self.halt_reason or "Orchestration halted before completion."
            console.print(f"[yellow]Goal incomplete. Halted: {reason}[/yellow]")

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)

        self.tasks = plan.tasks
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not self.tasks:
            if self.latest_test_result is not None and not self.latest_test_result.passed:
                self.halt_reason = "No tasks planned after failed verification."
            else:
                self.completed = True
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            workers = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in self.tasks
            ]
            results = await asyncio.gather(*workers, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        try:
            test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, self.tasks)
        except Exception as exc:
            test_result = TestResult(passed=False, summary=str(exc), failures=[str(exc)])
        self.latest_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            if any(task.status == "failed" for task in self.tasks):
                self.halt_reason = "Verification passed, but one or more tasks failed."
            elif all(task.status == "completed" for task in self.tasks):
                self.completed = True
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            failures = "\n".join(test_result.failures)
            self.goal = f"{self.original_goal}\n\nFix failing tests:\n{failures}"

    def _is_done(self) -> bool:
        return self.completed or (
            self.latest_test_result is not None
            and self.latest_test_result.passed
            and bool(self.tasks)
            and all(task.status == "completed" for task in self.tasks)
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
