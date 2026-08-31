from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        on_log: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.max_cycles = max_cycles
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.history: list[dict[str, Any]] = []
        self._on_log = on_log

    async def _log(self, message: str) -> None:
        console.print(message)
        if self._on_log:
            try:
                await self._on_log(message)
            except Exception:
                pass

    async def run(self) -> None:
        await self._log(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            await self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._log("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self._log(f"[yellow]Reached max cycles ({self.max_cycles}). Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        plan_context = self._build_plan_context()
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context=plan_context)
        self.current_plan = plan
        await self._log(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, cycle=self.cycles, goal=self.goal, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._log(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._log(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(goal=self.goal, plan=plan, client=self.client).run()

        if test_result.passed:
            await self._log(f"[green]Tests passed: {test_result.summary}[/green]")
            self.history.append({"cycle": self.cycles, "passed": True, "summary": test_result.summary})
        else:
            await self._log(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._log(f"  • {failure}")
            await self._log("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests from previous cycle:\n" + "\n".join(test_result.failures)
            self.history.append({"cycle": self.cycles, "passed": False, "failures": test_result.failures})

    def _build_plan_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "original_goal": self.original_goal,
            "current_goal": self.goal,
            "cycle": self.cycles,
            "previous_tasks": [t.model_dump() for t in self.current_plan.tasks] if self.current_plan else [],
            "history": self.history[-3:],
        }
        return context

    def _is_done(self) -> bool:
        if self.current_plan is None:
            return False
        completed = sum(1 for t in self.current_plan.tasks if t.status == "completed")
        failed = sum(1 for t in self.current_plan.tasks if t.status == "failed")
        if failed > 0:
            return False
        if len(self.current_plan.tasks) == 0:
            return True
        return completed >= len(self.current_plan.tasks)
