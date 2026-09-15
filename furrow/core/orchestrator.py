from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

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
        client_settings = getattr(client, "settings", None)
        self.settings = settings if settings is not None else client_settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max(0, self.settings.max_cycles)
        self.current_plan: Plan | None = None
        self.latest_test_result: TestResult | None = None
        self._failure_context = ""

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while self.max_cycles == 0 or self.cycles < self.max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        if self.max_cycles and self.cycles >= self.max_cycles and not self._is_done():
            console.print(f"[yellow]Maximum cycles ({self.max_cycles}) reached. Halting.[/yellow]")

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self._planning_goal())
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        worker_failures = []
        if plan.tasks:
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in plan.tasks
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    worker_failures.append(f"Task {task.id} failed: {result}")
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.latest_test_result = test_result

        failure_sections = []
        if worker_failures:
            failure_sections.append("Worker failures:\n" + "\n".join(worker_failures))
        if test_result.failures or not test_result.passed:
            test_failures = "\n".join(test_result.failures) or test_result.summary
            failure_sections.append("Latest test failures:\n" + test_failures)
            if not test_result.passed:
                console.print(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    console.print(f"  • {failure}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        self._failure_context = "\n\n".join(failure_sections)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")

    def _planning_goal(self) -> str:
        if not self._failure_context:
            return self.original_goal
        return f"{self.original_goal}\n\n{self._failure_context}"

    def _is_done(self) -> bool:
        if self.current_plan is None or self.latest_test_result is None:
            return False

        tasks = self.current_plan.tasks
        return all(task.status == "completed" for task in tasks) and self.latest_test_result.passed

    def _get_tasks(self) -> list[Any]:
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
