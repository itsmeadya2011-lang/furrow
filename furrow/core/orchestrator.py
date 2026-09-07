from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()

EventCallback = Callable[[dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self._last_plan: Plan | None = None
        self._history: list[dict[str, Any]] = []
        self._event_callback = event_callback

    def _emit(self, event: dict[str, Any]) -> None:
        if self._event_callback is not None:
            try:
                self._event_callback(event)
            except Exception:
                pass

    async def run(self) -> None:
        self._emit({"type": "start", "goal": self.goal})
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.settings and self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                msg = f"Reached max_cycles ({self.settings.max_cycles}). Halting."
                console.print(f"[yellow]{msg}[/yellow]")
                self._emit({"type": "stop", "reason": msg})
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit({"type": "cycle_start", "cycle": self.cycles})
            await self._cycle()
            if self._is_done():
                msg = "Goal complete. Halting."
                console.print(f"[bold green]{msg}[/bold green]")
                self._emit({"type": "done", "message": msg})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, history=self._history)
        self._last_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit({"type": "plan", "plan": plan.model_dump()})

        if not plan.tasks:
            msg = "No tasks planned. Goal may be complete."
            console.print(f"[yellow]{msg}[/yellow]")
            self._emit({"type": "no_tasks", "message": msg})
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_summary: list[dict[str, Any]] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit({"type": "task_failed", "task_id": task.id, "error": str(result)})
                cycle_summary.append({"id": task.id, "status": "failed", "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit({"type": "task_completed", "task_id": task.id, "result": result})
                cycle_summary.append({"id": task.id, "status": "completed", "result": result})

        self._history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "results": cycle_summary,
        })

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        self._emit({"type": "test_result", "passed": test_result.passed, "summary": test_result.summary, "failures": test_result.failures})
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        failed_tasks = [t for t in plan.tasks if t.status == "failed"]
        if failed_tasks:
            console.print(f"[yellow]{len(failed_tasks)} task(s) failed. Will retry in next cycle.[/yellow]")
            failed_descriptions = "\n".join(f"- {t.id}: {t.description}" for t in failed_tasks)
            self.goal = (
                "Retry previously failed tasks:\n" + failed_descriptions + "\n\n"
                "Also fix any test failures if present:\n" + "\n".join(test_result.failures or [])
            )

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        if self._last_plan is not None:
            return self._last_plan.tasks
        return []
