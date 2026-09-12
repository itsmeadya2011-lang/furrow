from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Awaitable

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
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        broadcast: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list = []
        self.broadcast = broadcast

    async def _emit(self, message: str) -> None:
        console.print(message)
        if self.broadcast:
            await self.broadcast(message)

    @property
    def persist_path(self) -> Path:
        return self.client.settings.persist_path

    async def save_state(self) -> None:
        path = self.persist_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "goal": self.goal,
            "tasks": [task.model_dump() for task in self._tasks],
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(state, file)

    async def load_state(self) -> bool:
        path = self.persist_path
        if not path.exists():
            return False

        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        self.goal = state["goal"]
        self._tasks = [TaskModel.model_validate(task) for task in state["tasks"]]
        return True

    async def run(self) -> None:
        await self.load_state()
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            await self.save_state()
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                await self._emit(f"[bold yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/bold yellow]")
                await self.save_state()
                break
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                await self.save_state()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        plan.tasks = sorted(plan.tasks, key=lambda task: len(task.dependencies))
        self._tasks = plan.tasks

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        task_by_id = {task.id: task for task in plan.tasks}
        ready_tasks = []
        for task in plan.tasks:
            dependencies_met = all(
                dependency in task_by_id and task_by_id[dependency].status == "completed"
                for dependency in task.dependencies
            )
            if dependencies_met:
                ready_tasks.append(task)
            else:
                task.status = "skipped"

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in ready_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(ready_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._tasks
