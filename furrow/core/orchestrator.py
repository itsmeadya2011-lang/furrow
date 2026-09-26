from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
import structlog

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import logger

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self.on_event = on_event

    async def _emit(self, message: str) -> None:
        logger.info("orchestrator_event", message=message)
        if self.on_event is not None:
            await self.on_event(message)

    async def run(self) -> None:
        await self._emit(f"Starting Furrow — Goal: {self.goal}")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"Cycle {self.cycles} starting")
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            # Check max_cycles (0 = infinite)
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._emit(f"Reached max_cycles ({self.settings.max_cycles}). Stopping.")
                console.print(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                break

            if self._is_done():
                await self._emit("Goal complete. Halting.")
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Persist tasks so _get_tasks() and _is_done() work across cycles
        self._tasks = plan.tasks

        if not plan.tasks:
            await self._emit("No tasks planned. Goal may be complete.")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._emit(f"Executing {len(plan.tasks)} task(s) in parallel (max {self.settings.max_parallel_tasks})")
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_worker(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"Task {task.id} failed: {result}")
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"Task {task.id} completed")
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"Tests passed: {test_result.summary}")
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"Tests failed: {test_result.summary}")
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

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

    def get_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of the orchestrator's current state."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self._tasks],
        }
