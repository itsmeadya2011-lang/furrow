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
from furrow.config import Settings, TaskModel, TaskStatus, TestResult
from furrow.core.state import save_state
from furrow.llm import LLMClient

console = Console()

EventCallback = Callable[[dict[str, Any]], Any]


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.history: list[dict[str, Any]] = []
        self._on_event = on_event
        self._stop_requested = False

    async def _emit(self, event: str, **data: Any) -> None:
        if self._on_event is None:
            return
        payload = {"event": event, **data}
        result = self._on_event(payload)
        if asyncio.iscoroutine(result):
            await result

    async def run(self) -> None:
        await self._emit("start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit("cycle", cycle=self.cycles)
            try:
                await self._cycle()
            except Exception as e:
                await self._emit("error", cycle=self.cycles, error=str(e))
                console.print(f"[red]Cycle {self.cycles} error: {e}[/red]")
                await self._finish("error")
                return
            self.history.append(
                {
                    "cycle": self.cycles,
                    "tasks": [t.model_dump() for t in self.tasks],
                    "test": self.last_test_result.model_dump() if self.last_test_result else None,
                }
            )
            save_state(self.state(), workspace=self.settings.workspace)
            done = self._should_stop()
            await self._emit("cycle_end", cycle=self.cycles, done=done)
            if self._stop_requested:
                console.print("[yellow]Stop requested. Halting.[/yellow]")
                await self._finish("stopped")
                return
            if done:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._finish("complete")
                return
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]")
                await self._finish("max_cycles")
                return

    async def _finish(self, reason: str) -> None:
        await self._emit("done", cycles=self.cycles, reason=reason)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.tasks = plan.tasks
        await self._emit(
            "plan",
            cycle=self.cycles,
            tasks=[t.model_dump() for t in plan.tasks],
            rationale=plan.rationale,
        )
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return
        chunk_size = max(1, self.settings.max_parallel_tasks)
        for chunk in _chunked(plan.tasks, chunk_size):
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                results = await asyncio.gather(
                    *[self._run_task(task) for task in chunk], return_exceptions=True
                )
            for task, result in zip(chunk, results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(
                    "task_result",
                    cycle=self.cycles,
                    task_id=task.id,
                    status=task.status.value,
                    result=task.result or "",
                )
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        self.last_test_result = test_result
        await self._emit(
            "tests",
            cycle=self.cycles,
            passed=test_result.passed,
            summary=test_result.summary,
            failures=test_result.failures,
        )
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = (
                f"Original goal: {self.original_goal}\n\nFix these failing tests:\n"
                + "\n".join(test_result.failures)
            )

    async def _run_task(self, task: TaskModel) -> str:
        task.status = TaskStatus.RUNNING
        await self._emit("task_start", cycle=self.cycles, task_id=task.id)
        return await WorkerAgent(task=task, client=self.client).run()

    def _should_stop(self) -> bool:
        if not self.tasks:
            return True
        return self.last_test_result is not None and self.last_test_result.passed

    def state(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "last_test": self.last_test_result.model_dump() if self.last_test_result else None,
            "status": "running",
        }

    def stop(self) -> None:
        self._stop_requested = True