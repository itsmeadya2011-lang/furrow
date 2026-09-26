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
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs Furrow's plan -> execute -> test loop until the goal is met."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        planner: PlannerAgent | None = None,
        tester: TesterAgent | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.planner = planner or PlannerAgent(client=self.client)
        self.tester = tester
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            done = await self._cycle()
            if done:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self._max_cycles_reached():
                console.print("[yellow]Maximum cycles reached. Stopping.[/yellow]")
                break

    async def _cycle(self) -> bool:
        plan: Plan | None = None
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.goal)
        except Exception as e:  # planning failure: cannot proceed this cycle
            console.print(f"[red]Planning failed: {e}[/red]")
            return True

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_result = None
            return True

        self.tasks = plan.tasks
        results = await self._run_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        tester = self.tester or TesterAgent(client=self.client)
        with Status("[bold yellow]Testing...", console=console):
            test_result = await tester.run(self.goal, plan.tasks)
        self.last_test_result = test_result

        self.history.append(
            {
                "cycle": self.cycles,
                "plan": plan.model_dump(),
                "results": [
                    {"id": t.id, "status": t.status, "result": t.result} for t in plan.tasks
                ],
                "test_result": test_result.model_dump(),
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            return True

        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  • {failure}")
        console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
        return False

    async def _run_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run(task: TaskModel):
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        return await asyncio.gather(*[_run(t) for t in tasks], return_exceptions=True)

    def _max_cycles_reached(self) -> bool:
        return self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles

    def _is_done(self) -> bool:
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed:
            return False
        if not self.tasks:
            return True
        if completed == len(self.tasks) and self.last_test_result and self.last_test_result.passed:
            return True
        return False
