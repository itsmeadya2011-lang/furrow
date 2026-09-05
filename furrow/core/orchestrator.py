from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        model_override: str | None = None,
        max_cycles: int | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.planner_produced_new_work = False

        self.settings = settings.model_copy()
        if model_override is not None:
            self.settings.model = model_override
        if max_cycles is not None:
            self.settings.max_cycles = max_cycles
        if workspace is not None:
            self.settings.workspace = workspace

        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.settings.max_cycles}). Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self.planner_produced_new_work = False
            try:
                await self._cycle()
            except Exception as e:
                console.print(f"[red]Cycle failed: {e}[/red]")
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
        except Exception as e:
            console.print(f"[red]Planner failed: {e}[/red]")
            return

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.planner_produced_new_work = True

        limited_tasks = plan.tasks[: self.settings.max_parallel_tasks]

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in limited_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(limited_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        self.tasks.extend(limited_tasks)

        try:
            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await TesterAgent(client=self.client).run(self.goal, limited_tasks)
        except Exception as e:
            console.print(f"[red]Testing failed: {e}[/red]")
            return

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"{self.original_goal}\nFix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks) and not self.planner_produced_new_work:
            return True
        return False
