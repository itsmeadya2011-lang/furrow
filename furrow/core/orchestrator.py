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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict], Any] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_tasks: list[Any] = []
        self._last_test_passed = True
        self._on_event = on_event

    async def _emit(self, event: dict) -> None:
        if self._on_event is not None:
            result = self._on_event(event)
            if asyncio.iscoroutine(result):
                await result

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        await self._emit({"type": "start", "goal": self.original_goal})

        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit({"type": "cycle", "number": self.cycles})
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit({"type": "complete", "goal": self.original_goal})
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[bold yellow]Reached max cycles ({max_cycles}). Stopping.[/bold yellow]")
                await self._emit({"type": "max_cycles", "limit": max_cycles, "cycles": self.cycles})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit({"type": "plan", "plan": plan.model_dump()})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit({"type": "no_tasks"})
            if not self._last_test_passed:
                console.print("[red]Tests previously failed but planner returned no tasks. Stopping.[/red]")
                await self._emit({"type": "stuck", "reason": "planner returned no tasks after test failure"})
            return

        self._current_tasks = plan.tasks

        max_parallel = max(1, self.client.settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_one(task: Any) -> tuple[Any, Any]:
            async with semaphore:
                result = await WorkerAgent(task=task, client=self.client).run()
                return task, result

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            wrapped = [_run_one(task) for task in plan.tasks]
            results = await asyncio.gather(*wrapped, return_exceptions=True)

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit({"type": "task_failed", "task_id": task.id, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit({"type": "task_completed", "task_id": task.id, "summary": result})

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._last_test_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit({"type": "test_passed", "summary": test_result.summary})
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            await self._emit({"type": "test_failed", "summary": test_result.summary, "failures": test_result.failures})
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Preserve original goal; use a working goal for fix attempts
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self._current_tasks:
            return True
        completed = sum(1 for t in self._current_tasks if t.status == "completed")
        failed = sum(1 for t in self._current_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed < len(self._current_tasks):
            return False
        return self._last_test_passed

    def _get_tasks(self) -> list[Any]:
        return self._current_tasks