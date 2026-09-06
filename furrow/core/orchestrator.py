from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_cycle: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings if settings is not None else default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.on_cycle = on_cycle
        self.cycles = 0
        self.current_plan: Plan | None = None
        # Track the most recently executed tasks so completion can be evaluated
        self.tasks: list[TaskModel] = []
        # Track the most recent test result; a failing test run means the goal
        # is not complete and the loop should keep iterating to fix issues.
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            # No tasks means nothing left to do -> treat as complete
            self.tasks = []
            return

        # Enforce max_parallel_tasks with a semaphore
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        # Keep tasks across cycles so completion is measurable
        self.tasks = list(plan.tasks)
        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        # Emit a plain-text summary for consumers (e.g. the web UI). A failure
        # here (e.g. a disconnected websocket) must not abort the development loop.
        if self.on_cycle is not None:
            try:
                await self.on_cycle(self._cycle_summary())
            except Exception:
                pass

    def _cycle_summary(self) -> str:
        lines = [f"Cycle {self.cycles}:"]
        if not self.tasks:
            lines.append("  no tasks planned")
        for t in self.tasks:
            lines.append(f"  - Task {t.id} [{t.status}]: {t.description}")
        if self.last_test_result is not None:
            verdict = "passed" if self.last_test_result.passed else "failed"
            lines.append(f"  Tests {verdict}: {self.last_test_result.summary}")
        return "\n".join(lines)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        # No tasks to work on -> nothing left to do.
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        # Any failed task means we need another cycle to retry.
        if failed > 0:
            return False
        # Tasks not yet completed (still pending/running) -> keep going.
        if completed < len(tasks):
            return False
        # All tasks completed: only done if the latest test run also passed.
        if self.last_test_result is not None and not self.last_test_result.passed:
            return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    # -- State persistence / recovery -------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "current_plan": self.current_plan.model_dump() if self.current_plan else None,
            "tasks": [t.model_dump() for t in self.tasks],
            "last_test_result": (
                self.last_test_result.model_dump() if self.last_test_result else None
            ),
        }

    def save_state(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state_dict(), indent=2, default=str))

    @classmethod
    def load_state(cls, path: str | Path, client: LLMClient | None = None) -> Orchestrator:
        p = Path(path)
        data = json.loads(p.read_text())
        plan = Plan(**data["current_plan"]) if data.get("current_plan") else None
        tasks = [TaskModel(**t) for t in data.get("tasks", [])]
        orch = cls(goal=data["goal"], client=client)
        orch.cycles = data.get("cycles", 0)
        orch.current_plan = plan
        orch.tasks = tasks
        if data.get("last_test_result"):
            orch.last_test_result = TestResult(**data["last_test_result"])
        return orch
