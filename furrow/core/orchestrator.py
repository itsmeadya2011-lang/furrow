from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
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
EventCallback = Callable[[dict[str, Any]], Any]  # may be sync or async


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
        # Pass settings only when provided; otherwise let LLMClient fall back to
        # its module-default Settings (avoids passing settings=None, which would
        # override the default with None).
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings: Settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.tasks: list[TaskModel] = []          # every task seen, across cycles
        self.test_results: list[TestResult] = []  # one per cycle that ran tests
        self.last_test_result: TestResult | None = None
        self._event_handlers: list[EventCallback] = []
        if on_event is not None:
            self._event_handlers.append(on_event)

    def on(self, callback: EventCallback) -> None:
        """Register an additional event handler (sync or async)."""
        self._event_handlers.append(callback)

    async def emit(self, event: str, **data: Any) -> None:
        """Emit an event payload to all registered handlers.

        Handlers may be synchronous or asynchronous; awaitable return values are
        awaited before moving on to the next handler.
        """
        payload: dict[str, Any] = {"event": event, "cycle": self.cycles}
        payload.update(data)
        for handler in self._event_handlers:
            result = handler(payload)
            if inspect.isawaitable(result):
                await result

    async def run(self) -> None:
        """Run the plan -> execute -> test loop until the goal is complete.

        Prints the Furrow banner, then iterates cycles (incrementing ``cycles``,
        emitting ``cycle_start`` / ``cycle_end``). Stops when ``_is_done()``
        reports completion, or when ``settings.max_cycles`` (> 0) is reached.
        """
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self.emit("start", goal=self.goal)
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self.emit("cycle_start")
            try:
                await self._cycle()
            except Exception as exc:
                await self.emit("error", error=str(exc))
                raise
            await self.emit("cycle_end")
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self.emit("complete")
                break
            elif self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]"
                )
                await self.emit("max_cycles_reached", max_cycles=self.settings.max_cycles)
                break

    async def _cycle(self) -> None:
        """Execute one cycle: plan, run workers in parallel, then test."""
        # 1. Plan
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        self.tasks.extend(plan.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self.emit("plan", plan=plan.model_dump())

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self.emit("no_tasks")
            return

        # 2. Execute workers in parallel, bounded by a semaphore
        sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

        async def _run_task(task: TaskModel) -> Any:
            async with sem:
                await self.emit("task_start", task=task.model_dump())
                console.print(f"[yellow]Starting task {task.id}[/yellow]")
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self.emit("task_failed", task=task.model_dump(), error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self.emit("task_complete", task=task.model_dump())

        # 3. Test
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result
        self.test_results.append(test_result)
        await self.emit("test_result", result=test_result.model_dump())

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = self._fix_goal(test_result)
            await self.emit("fix_cycle", failures=test_result.failures)

    def _fix_goal(self, test_result: TestResult) -> str:
        """Compose a new goal asking the planner to fix the test failures."""
        failures = "\n".join(f"- {f}" for f in test_result.failures) or test_result.summary
        return (
            f"{self.original_goal}\n\n"
            "The previous cycle's tests failed. Fix these failures while still "
            f"serving the original goal:\n{failures}"
        )

    def _is_done(self) -> bool:
        """Return whether the orchestrator has reached a terminal state.

        A terminal state is reached when the current plan has no pending/failed
        tasks and the most recent test result (if any) passed.
        """
        if self.plan is None:
            return False
        if not self.plan.tasks:
            return True
        if any(t.status in ("pending", "failed") for t in self.plan.tasks):
            return False
        return self.last_test_result is not None and self.last_test_result.passed

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the orchestration session."""
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        pending = sum(1 for t in self.tasks if t.status == "pending")
        tests_passed = sum(1 for t in self.test_results if t.passed)
        return {
            "goal": self.original_goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "tests_passed": tests_passed,
            "last_test": self.last_test_result.model_dump() if self.last_test_result else None,
        }
