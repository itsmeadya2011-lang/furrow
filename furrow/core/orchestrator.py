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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()

EventCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._on_event = on_event

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            await self._on_event(event)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit({"type": "start", "data": {"goal": self.goal}})
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                await self._emit({"type": "done", "data": {"reason": "max_cycles", "cycles": self.cycles}})
                break
            try:
                stop = await self._cycle()
            # Broad catch: a cycle error should halt the run rather than crash it.
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Cycle error: {exc}[/red]")
                await self._emit({"type": "error", "data": {"error": str(exc)}})
                break
            if stop or self._is_done():
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    await self._emit({"type": "done", "data": {"reason": "complete", "cycles": self.cycles}})
                break

    async def _cycle(self) -> bool:
        """Run a single cycle. Returns True if the orchestrator should stop."""
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.goal)
        # Broad catch: a planner error should halt the run rather than crash it.
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Planner error: {exc}[/red]")
            return True

        self._current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit({"type": "plan", "data": {"cycle": self.cycles, "plan": plan.model_dump()}})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit({"type": "cycle_complete", "data": {"cycle": self.cycles, "tasks": 0}})
            return False

        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        await self._emit({"type": "task_started", "data": {"cycle": self.cycles, "task_ids": [t.id for t in plan.tasks]}})

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(run_task(task) for task in plan.tasks), return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit({"type": "task_completed", "data": {"cycle": self.cycles, "task_id": task.id, "status": "failed", "result": str(result)}})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit({"type": "task_completed", "data": {"cycle": self.cycles, "task_id": task.id, "status": "completed", "result": result}})

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        await self._emit({"type": "test_result", "data": {"cycle": self.cycles, "passed": test_result.passed, "summary": test_result.summary, "failures": test_result.failures}})

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        await self._emit({"type": "cycle_complete", "data": {"cycle": self.cycles}})
        return False

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(t.status == "failed" for t in tasks):
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[Any]:
        return self._current_plan.tasks if self._current_plan else []
