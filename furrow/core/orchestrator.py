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
from furrow.llm import LLMClient

_console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: dict[str, Any] = {}
        self._on_output = on_output
        self.console = _console

    def _emit(self, text: str) -> None:
        if self._on_output is not None:
            result = self._on_output(text)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)

    async def run(self) -> None:
        self._emit(f"Furrow starting. Goal: {self.goal}")
        while True:
            self.cycles += 1
            self._emit(f"--- Cycle {self.cycles} ---")
            await self._cycle()
            if self._is_done():
                self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(f"Plan: {plan.rationale}")

        if not plan.tasks:
            self._emit("No tasks planned. Goal may be complete.")
            return

        for task in plan.tasks:
            self._tasks[task.id] = task

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            sem = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

            async def _run(task: Any) -> Any:
                async with sem:
                    return await WorkerAgent(task=task, client=self.client).run()

            tasks = [_run(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                self._emit(f"Task {task.id} completed: {result}")

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit(f"Tests passed: {test_result.summary}")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self._emit(f"Tests failed: {test_result.summary}")
            self._emit("Will attempt fix in next cycle.")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.client.settings.max_cycles and self.cycles >= self.client.settings.max_cycles:
            self.console.print(
                f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]"
            )
            return True

        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return list(self._tasks.values())