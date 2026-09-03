from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Optional, Union

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

EventHandler = Optional[Callable[[dict[str, Any]], Union[None, Awaitable[None]]]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventHandler = None,
    ) -> None:
        self.goal = goal
        self.original_goal: str = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._plan: Plan | None = None
        self._task_statuses: dict[str, str] = {}
        self._last_failures: list[str] = []
        self._last_passed: bool = False
        self.on_event: EventHandler = on_event

    async def _emit(self, event: dict[str, Any]) -> None:
        cb = self.on_event
        if cb is None:
            return
        if inspect.iscoroutinefunction(cb):
            await cb(event)
        else:
            result = cb(event)
            if inspect.iscoroutine(result):
                await result

    def _run_with_semaphore(
        self, sem: asyncio.Semaphore, agent: WorkerAgent
    ) -> asyncio.Future[str]:
        async def _go() -> str:
            async with sem:
                return await agent.run()

        return asyncio.ensure_future(_go())

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit(
                    {
                        "type": "done",
                        "reason": "max_cycles_reached",
                        "cycle": self.cycles,
                    }
                )
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            self.cycles += 1

            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(
                {
                    "type": "cycle_start",
                    "cycle": self.cycles,
                    "goal": self.original_goal,
                }
            )

            done_reason = await self._cycle()
            await self._emit({"type": "cycle_end", "cycle": self.cycles})

            if done_reason is not None:
                await self._emit({"type": "done", "reason": done_reason})
                if done_reason == "complete":
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                else:
                    console.print(
                        f"[yellow]Halting: {done_reason}[/yellow]"
                    )
                break

            if self._is_done():
                await self._emit({"type": "done", "reason": "complete"})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> Optional[str]:
        planner_goal = self.original_goal
        if self._last_failures:
            planner_goal = (
                self.original_goal
                + "\n\nPreviously failed tests:\n"
                + "\n".join(self._last_failures)
            )

        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(planner_goal)
        except Exception as exc:
            console.print(f"[red]Planner failed: {exc}[/red]")
            return "planner_failed"

        self._plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit({"type": "plan", "plan": plan.model_dump()})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return None

        sem = asyncio.Semaphore(self.client.settings.max_parallel_tasks)
        for task in plan.tasks:
            await self._emit({"type": "task_started", "task_id": task.id})

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                self._run_with_semaphore(sem, WorkerAgent(task=task, client=self.client))
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._task_statuses[task.id] = "failed"
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(
                    {"type": "task_failed", "task_id": task.id, "error": str(result)}
                )
            else:
                task.status = "completed"
                task.result = result
                self._task_statuses[task.id] = "completed"
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(
                    {"type": "task_completed", "task_id": task.id, "result": result}
                )

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result: TestResult = await TesterAgent(client=self.client).run(
                self.original_goal, plan.tasks
            )

        self._last_failures = list(test_result.failures)
        self._last_passed = test_result.passed
        await self._emit(
            {
                "type": "test_result",
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

        return None

    def _is_done(self) -> bool:
        if self._plan is None:
            return True
        if not self._last_passed:
            return False
        tasks = self._plan.tasks
        if not tasks:
            return True
        for t in tasks:
            status = self._task_statuses.get(t.id, t.status)
            if status != "completed":
                return False
        return True

    def _get_tasks(self) -> list[Any]:
        return list(self._plan.tasks) if self._plan else []
