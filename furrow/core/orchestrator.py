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
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.fix_goal: str | None = None
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tasks))
        self._on_event = on_event

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                self.cycles += 1
                self._emit({"type": "cycle", "cycle": self.cycles})
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
                if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                    console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                    break
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted. Halting.[/yellow]")
            raise
        finally:
            self._emit({"type": "done", "cycles": self.cycles})

    async def _cycle(self) -> None:
        goal = self.original_goal
        if self.fix_goal:
            goal = f"{self.original_goal}\n\n{self.fix_goal}"
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit({"type": "plan", "plan": plan.model_dump()})

        self._tasks = list(plan.tasks)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[self._run_task(task) for task in plan.tasks],
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            self._emit({"type": "task", "task": task.id, "status": task.status, "result": task.result})

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._emit({"type": "test", "result": test_result.model_dump()})

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.fix_goal = None
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.fix_goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
