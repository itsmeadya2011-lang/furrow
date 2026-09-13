from __future__ import annotations

import asyncio
import uuid
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Session, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[Any] = []
        self._plan: Plan | None = None
        self._session_id = str(uuid.uuid4())
        self._session = self.client.settings.load_session()
        if self._session and self._session.goal == goal and self._session.cycles > 0:
            self.cycles = self._session.cycles
            self.goal = self._session.goal
            if self._session.plan:
                self._plan = self._session.plan
                self._tasks = list(self._session.plan.tasks)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        while max_cycles == 0 or self.cycles < max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_session()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
        else:
            if max_cycles > 0:
                console.print(f"[yellow]Reached max cycles ({max_cycles}). Halting.[/yellow]")

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._tasks = list(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self._tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self._tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self._tasks:
            return True
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._tasks):
            return True
        return False

    def _save_session(self) -> None:
        plan = self._plan or Plan(tasks=self._tasks, rationale="")
        session = Session(
            session_id=self._session_id,
            goal=self.goal,
            plan=plan,
            cycles=self.cycles,
        )
        self.client.settings.save_session(session)

    def _get_tasks(self) -> list[Any]:
        return list(self._tasks)
