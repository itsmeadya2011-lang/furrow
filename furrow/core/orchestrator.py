"""Orchestrator module that coordinates planning, execution, and testing cycles."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str, dict], Awaitable[None] | None] | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._last_test_result: TestResult | None = None
        self._retry_summary: str | None = None
        self._done_reason: str | None = None
        self._stop_event = stop_event or asyncio.Event()
        self.on_event = on_event

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if self._stop_event.is_set():
                self._done_reason = "stopped"
                console.print("[yellow]Stop requested. Halting.[/yellow]")
                await self._emit("done", {"reason": "stopped"})
                break

            self.cycles += 1
            await self._emit("cycle_start", {"cycle": self.cycles, "goal": self.goal})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._stop_event.is_set():
                self._done_reason = "stopped"
                console.print("[yellow]Stop requested. Halting.[/yellow]")
                await self._emit("done", {"reason": "stopped"})
                break

            done = self._is_done()
            await self._emit("cycle_end", {"cycle": self.cycles, "is_done": done})

            if done:
                self._done_reason = "complete"
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("done", {"reason": "complete"})
                break

            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self._done_reason = "max_cycles"
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                await self._emit("done", {"reason": "max_cycles"})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, failure_context=self._retry_summary)
        self._current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan_ready", {"plan": plan.model_dump()})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._last_test_result = None
            return

        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
        completed_tasks: set[str] = set()
        failed_tasks: set[str] = set()

        waves = self._build_waves(plan.tasks)
        for wave in waves:
            if self._stop_event.is_set():
                break

            wave_tasks = [
                task for task in plan.tasks
                if task.id in wave and task.id not in completed_tasks and task.id not in failed_tasks
            ]

            async def _run_with_semaphore(task: Any) -> None:
                if self._stop_event.is_set():
                    return
                await self._emit("task_start", {"id": task.id, "description": task.description})
                async with semaphore:
                    if self._stop_event.is_set():
                        return
                    try:
                        result = await WorkerAgent(task=task, client=self.client).run()
                        task.status = "completed"
                        task.result = result
                        console.print(f"[green]Task {task.id} completed[/green]")
                        await self._emit(
                            "task_complete",
                            {"id": task.id, "status": "completed", "result": result},
                        )
                    except Exception as e:
                        task.status = "failed"
                        task.result = str(e)
                        failed_tasks.add(task.id)
                        console.print(f"[red]Task {task.id} failed: {e}[/red]")
                        await self._emit(
                            "task_failed",
                            {"id": task.id, "status": "failed", "error": str(e)},
                        )

            await asyncio.gather(*[_run_with_semaphore(task) for task in wave_tasks])

            for task in wave_tasks:
                if task.status == "completed":
                    completed_tasks.add(task.id)
                elif task.status == "failed":
                    failed_tasks.add(task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._last_test_result = test_result
        await self._emit(
            "test_complete",
            {
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
            },
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._retry_summary = None
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._retry_summary = f"Tests failed:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        plan = self._current_plan
        if plan is None:
            return False

        tasks = plan.tasks
        if not tasks:
            return False

        all_completed = all(t.status == "completed" for t in tasks)
        any_failed = any(t.status == "failed" for t in tasks)

        if any_failed:
            return False

        if not all_completed:
            return False

        if self._last_test_result is None:
            return False

        return self._last_test_result.passed

    def _get_tasks(self) -> list[Any]:
        if self._current_plan is None:
            return []
        return self._current_plan.tasks

    def _build_waves(self, tasks: list[Any]) -> list[set[str]]:
        task_map = {task.id: task for task in tasks}
        remaining = {task.id for task in tasks}
        waves: list[set[str]] = []

        while remaining:
            wave: set[str] = set()
            for task_id in list(remaining):
                task = task_map[task_id]
                deps = set(task.dependencies)
                if deps.issubset({t for w in waves for t in w}) or not deps:
                    wave.add(task_id)

            if not wave:
                break

            waves.append(wave)
            remaining -= wave

        return waves

    async def _emit(self, event: str, data: dict) -> None:
        if self.on_event is None:
            return
        try:
            result = self.on_event(event, data)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass
