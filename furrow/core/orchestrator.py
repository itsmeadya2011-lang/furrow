from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str, dict], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._all_tasks: list[TaskModel] = []
        self._on_event = on_event

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("done", {"reason": "goal_complete"})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit("done", {"reason": "max_cycles_reached", "cycles": self.cycles})
                console.print(f"[yellow]Halting after {self.cycles} cycle(s) (max_cycles={max_cycles}).[/yellow]")
                break

    async def _cycle(self) -> None:
        await self._emit("cycle_start", {"cycle": self.cycles})
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit("plan", plan.model_dump())
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            settings = self.client.settings
            max_parallel = settings.max_parallel_tasks
            if max_parallel > 0:
                semaphore = asyncio.Semaphore(max_parallel)

                async def _run_with_semaphore(task: TaskModel) -> str:
                    async with semaphore:
                        return await WorkerAgent(task=task, client=self.client).run()

                tasks = [_run_with_semaphore(task) for task in plan.tasks]
            else:
                tasks = [WorkerAgent(task=task, client=self.client).run() for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            await self._emit(
                "task_done",
                {"id": task.id, "status": task.status, "result": task.result},
            )

        self._all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        await self._emit("tests", test_result.model_dump())

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            previous_goal = self.goal
            self.goal = (
                f"Fix failing tests:\n" + "\n".join(test_result.failures) + f"\n\nOriginal goal:\n{previous_goal}"
            )

    async def _emit(self, event: str, payload: dict) -> None:
        if self._on_event is not None:
            try:
                await self._on_event(event, payload)
            except Exception:
                pass

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._all_tasks if t.status == "completed")
        failed = sum(1 for t in self._all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._all_tasks):
            return True
        return False
