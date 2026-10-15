from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_progress: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self.on_progress = on_progress

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit("started", goal=self.goal)
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                self._emit("halted", reason="max_cycles", cycles=self.cycles)
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle_start", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("completed", cycles=self.cycles)
                break

    def _emit(self, event: str, **data: Any) -> None:
        if self.on_progress is not None:
            payload = {"event": event, **data}
            if asyncio.iscoroutinefunction(self.on_progress):
                asyncio.create_task(self.on_progress(payload))
            else:
                self.on_progress(payload)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit("no_tasks")
            return

        # Merge new tasks with existing tracked tasks (avoid duplicates by id)
        existing_ids = {t.id for t in self._tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self._tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            self._emit("tests_failed", summary=test_result.summary, failures=test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks


__all__ = ["Orchestrator"]
