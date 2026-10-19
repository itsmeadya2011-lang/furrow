from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()

# Type for event callbacks: async function that takes an event dict
EventCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.original_goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self._on_event: EventCallback | None = None

    def on_event(self, callback: EventCallback | None) -> None:
        """Set a callback to receive orchestrator events."""
        self._on_event = callback

    async def _emit(self, event_type: str, **kwargs: Any) -> None:
        """Emit an event to the callback if set."""
        event: dict[str, Any] = {"type": event_type, "cycle": self.cycles, **kwargs}
        if self._on_event is not None:
            try:
                await self._on_event(event)
            except Exception:  # noqa: BLE001, S110 - don't let event callbacks break the orchestrator
                pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit("start", goal=self.goal)
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit("cycle_start", cycle=self.cycles)

            # Enforce max_cycles limit
            max_cycles = self.settings.max_cycles
            if max_cycles > 0 and self.cycles > max_cycles:
                console.print(f"[bold yellow]Reached max cycles limit ({max_cycles}). Stopping.[/bold yellow]")
                await self._emit("max_cycles_reached", max_cycles=max_cycles)
                break

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("done")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            self.plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(self.plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan", plan=self.plan.model_dump())

        if not self.plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("no_tasks")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit("task_completed", task_id=task.id, result=result)

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(self.goal, self.plan.tasks)

        if self.last_test_result.passed:
            console.print(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
            await self._emit("tests_passed", summary=self.last_test_result.summary)
        else:
            console.print(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
            for failure in self.last_test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._emit(
                "tests_failed",
                summary=self.last_test_result.summary,
                failures=self.last_test_result.failures,
            )
            self.goal = "Fix failing tests:\n" + "\n".join(self.last_test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            # No tasks planned - goal may be complete
            return self.last_test_result is not None and self.last_test_result.passed

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            return False

        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        if self.plan is not None:
            return self.plan.tasks
        return []