from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.stream_callback: Callable[[str, dict], None] | None = None

    def set_stream_callback(self, cb: Callable[[str, dict], None]) -> None:
        self.stream_callback = cb

    def _emit(self, event_type: str, data: dict) -> None:
        if self.stream_callback:
            self.stream_callback(event_type, data)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit("cycle_start", {"cycle": self.cycles, "goal": self.goal})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("goal_complete", {"cycles": self.cycles, "tasks": [t.model_dump() for t in self.tasks]})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        self._emit("planning", {"goal": self.goal, "cycle": self.cycles})
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit("plan_created", {"plan": plan.model_dump(), "cycle": self.cycles})
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks
        self._emit("task_executing", {"tasks": [t.model_dump() for t in self.tasks], "cycle": self.cycles})
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
                self._emit("task_failed", {"task": task.model_dump(), "error": str(result)})
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit("task_completed", {"task": task.model_dump(), "result": result})
                console.print(f"[green]Task {task.id} completed[/green]")

        self._emit("testing", {"goal": self.goal, "tasks": [t.model_dump() for t in self.tasks], "cycle": self.cycles})
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit("tests_passed", {"summary": test_result.summary, "cycle": self.cycles})
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit("tests_failed", {"summary": test_result.summary, "failures": test_result.failures, "cycle": self.cycles})
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            for task in self.tasks:
                task.status = "pending"
                task.result = None

    def _is_done(self) -> bool:
        max_cycles = settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return True
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
