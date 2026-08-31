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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles > self.client.settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)

        ready, pending = self._partition_tasks(self.tasks)
        console.print(f"Ready: {len(ready)}, Pending (deps): {len(pending)}")

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

            async def _run(task: TaskModel) -> None:
                async with semaphore:
                    agent = WorkerAgent(task=task, client=self.client, workspace=self.client.settings.workspace)
                    try:
                        result = await agent.run()
                        task.status = "completed"
                        task.result = result
                        console.print(f"[green]Task {task.id} completed[/green]")
                    except Exception as exc:
                        task.status = "failed"
                        task.result = str(exc)
                        console.print(f"[red]Task {task.id} failed: {exc}[/red]")

            await asyncio.gather(*(_run(t) for t in ready))

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _partition_tasks(self, tasks: list[TaskModel]) -> tuple[list[TaskModel], list[TaskModel]]:
        completed = {t.id for t in tasks if t.status == "completed"}
        failed = {t.id for t in tasks if t.status == "failed"}
        ready = []
        pending = []
        for t in tasks:
            if t.status in ("completed", "failed"):
                continue
            deps = set(t.dependencies or [])
            if deps.issubset(completed):
                ready.append(t)
            else:
                pending.append(t)
        return ready, pending

    def _is_done(self) -> bool:
        ready, pending = self._partition_tasks(self.tasks)
        if ready:
            return False
        if pending:
            return False
        if not self.tasks:
            return True
        return all(t.status == "completed" for t in self.tasks)
