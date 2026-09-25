from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else Settings())
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.current_plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self.on_event = on_event

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            await self.on_event(event)

    async def run(self) -> None:
        await self._emit({"type": "goal", "goal": self.goal})
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                await self._emit({"type": "max_cycles_reached", "cycles": self.cycles})
                break
            await self._emit({"type": "cycle_start", "cycle": self.cycles})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as e:  # noqa: BLE001
                await self._emit({"type": "error", "message": str(e), "cycle": self.cycles})
                console.print(f"[red]Cycle error: {e}[/red]")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit({"type": "done", "cycles": self.cycles, "summary": self.goal})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        await self._emit({"type": "plan", "plan": plan.model_dump(), "cycle": self.cycles})
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.all_tasks.extend(plan.tasks)

        if not plan.tasks:
            await self._emit({"type": "no_tasks", "cycle": self.cycles})
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

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
                await self._emit(
                    {"type": "task_failed", "task_id": task.id, "error": str(result), "cycle": self.cycles}
                )
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(
                    {"type": "task_completed", "task_id": task.id, "result": result, "cycle": self.cycles}
                )
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.last_test_result = test_result
        await self._emit(
            {
                "type": "test_result",
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
                "cycle": self.cycles,
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            await self._emit({"type": "max_cycles_reformed", "goal": self.goal, "cycle": self.cycles})

    def _is_done(self) -> bool:
        if self.current_plan is not None and len(self.current_plan.tasks) == 0:
            return True
        return self.last_test_result is not None and self.last_test_result.passed
