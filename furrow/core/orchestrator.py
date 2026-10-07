from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.on_event = on_event
        self.semaphore = asyncio.Semaphore(
            self.client.settings.max_parallel_tasks
        )
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _emit(self, event: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001, S110
                pass

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            if self._stop_event.is_set():
                self._emit({"type": "done", "reason": "stopped"})
                console.print("[bold yellow]Stopped by user. Halting.[/bold yellow]")
                return
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._stop_event.is_set():
                self._emit({"type": "done", "reason": "stopped"})
                console.print(
                    "[bold yellow]Stopped by user. Halting.[/bold yellow]"
                )
                return
            if self._is_done():
                self._emit({"type": "done", "reason": "complete"})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if (
                self.client.settings.max_cycles > 0
                and self.cycles >= self.client.settings.max_cycles
            ):
                self._emit({"type": "done", "reason": "cycle limit reached"})
                console.print(
                    "[bold yellow]Cycle limit reached. Halting.[/bold yellow]"
                )
                break

    async def _cycle(self) -> None:
        self._emit({"type": "cycle", "cycle": self.cycles})
        with Status("[bold yellow]Planning...", console=console) as _status:
            plan = await self.planner.plan(self.goal)
        self._emit({"type": "plan", "plan": plan.model_dump()})
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit({"type": "done", "reason": "no tasks"})
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks

        async def _run_worker(task: TaskModel):
            async with self.semaphore:
                self._emit(
                    {
                        "type": "task",
                        "task": task.model_dump(),
                        "status": "started",
                    }
                )
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(
                    {
                        "type": "task",
                        "task": task.model_dump(),
                        "status": "failed",
                    }
                )
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = str(result)
                self._emit(
                    {
                        "type": "task",
                        "task": task.model_dump(),
                        "status": "completed",
                    }
                )
                console.print(f"[green]Task {task.id} completed[/green]")

        if self._stop_event.is_set():
            return

        self._emit({"type": "log", "message": "Testing..."})
        with Status("[bold yellow]Testing...", console=console) as _status:
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )
        self._emit({"type": "test", "result": test_result.model_dump()})

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        return failed == 0 and completed >= len(self._get_tasks())

    def _get_tasks(self) -> list[Any]:
        return self.tasks