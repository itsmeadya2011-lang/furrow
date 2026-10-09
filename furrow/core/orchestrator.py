from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
log = get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        on_event: Callable[[str, Any], Any] | None = None,
        planner_factory: Callable[..., PlannerAgent] | None = None,
        worker_factory: Callable[..., Any] | None = None,
        tester_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.max_cycles = max_cycles
        self.on_event = on_event
        self._planner_factory = planner_factory
        self._worker_factory = worker_factory
        self._tester_factory = tester_factory

        self.planner = (planner_factory or PlannerAgent)(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self._last_cycle_had_tasks = True

    async def _emit(self, event: str, data: Any = None) -> None:
        if self.on_event is not None:
            try:
                result = self.on_event(event, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.warning("event_handler_failed", event_name=event)

    async def run(self) -> None:
        log.info("run_started", goal=self.goal)
        await self._emit("start", {"goal": self.goal})
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit("cycle_start", {"cycle": self.cycles})
            await self._cycle()
            if self.max_cycles and self.cycles >= self.max_cycles:
                console.print(f"[bold yellow]Reached max cycles ({self.max_cycles}). Halting.[/bold yellow]")
                await self._emit("max_cycles_reached", {"cycles": self.cycles})
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("done", {"cycles": self.cycles, "tasks": len(self.tasks)})
                break
            if self._stalled():
                console.print("[bold red]Stalled: no new tasks planned while failures remain. Halting.[/bold red]")
                await self._emit("stalled", {"cycles": self.cycles, "tasks": len(self.tasks)})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        log.info("plan_ready", tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan", plan.model_dump())

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("no_tasks", {})
            self._last_cycle_had_tasks = False
            if self.last_test_result is not None and not self.last_test_result.passed:
                console.print("[bold red]Tests failing but no new tasks planned. Halting.[/bold red]")
                await self._emit("stalled", {"reason": "no_tasks_with_failures"})
            return

        self._last_cycle_had_tasks = True

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [self._run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results, strict=True):
            if isinstance(result, Exception):
                task.status = "failed"
                error_msg = str(result)
                task.result = error_msg
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("task_failed", task_id=task.id, error=error_msg)
                await self._emit("task_failed", {"task_id": task.id, "error": error_msg})
            else:
                task.status = "completed"
                task.result = str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                log.info("task_completed", task_id=task.id)
                await self._emit("task_completed", {"task_id": task.id, "summary": result})
            self.tasks.append(task)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self._run_tester(self.goal, plan.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", summary=test_result.summary)
            await self._emit("tests_passed", test_result.model_dump())
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            log.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            await self._emit("tests_failed", test_result.model_dump())
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, task: TaskModel) -> str:
        from furrow.agents.worker import WorkerAgent

        factory = self._worker_factory or WorkerAgent
        worker = factory(task, self.client)
        return await worker.run()

    async def _run_tester(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        from furrow.agents.tester import TesterAgent

        factory = self._tester_factory or TesterAgent
        tester = factory(self.client)
        return await tester.run(goal, tasks)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if self.last_test_result is not None and not self.last_test_result.passed:
            return False
        return completed >= len(self.tasks)

    def _stalled(self) -> bool:
        """True when no forward progress can be made.

        Only triggers when the last cycle produced no new tasks AND there are
        unresolved failures (task failures or failing tests).
        """
        if self._last_cycle_had_tasks or not self.tasks:
            return False
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return True
        return self.last_test_result is not None and not self.last_test_result.passed
