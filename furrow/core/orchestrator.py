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
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_failures: list[str] = []
        self.completed_summaries: list[str] = []
        self.on_event = on_event
        self._last_plan_task_descriptions: list[str] = []
        self._stop_reason: str | None = None

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(payload)
        except Exception:  # noqa: BLE001 - a broken listener must not break the loop
            pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            max_cycles = int(getattr(self.client.settings, "max_cycles", 0) or 0)
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._stop_reason = "max cycles reached"
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit({"type": "cycle_start", "cycle": self.cycles})
            await self._cycle()
            self._emit({"type": "cycle_end", "cycle": self.cycles})
            if self._is_done():
                if self._stop_reason is None:
                    self._stop_reason = "complete"
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    def _build_goal_context(self) -> str:
        parts: list[str] = []
        parts.append(f"Original goal: {self.original_goal}")
        if self.completed_summaries:
            completed_desc = "\n".join(f"- {s}" for s in self.completed_summaries[-20:])
            parts.append(f"Completed tasks:\n{completed_desc}")
        if self.last_failures:
            failures = "\n".join(f"- {f}" for f in self.last_failures[-20:])
            parts.append(f"Last test failures:\n{failures}")
        return "\n\n".join(parts)

    async def _cycle(self) -> None:
        goal_context = self._build_goal_context()
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(goal_context)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit({"type": "plan", "cycle": self.cycles, "tasks": [t.model_dump() for t in plan.tasks], "rationale": plan.rationale})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._last_plan_task_descriptions = []
            self._stop_reason = "no tasks"
            return

        current_descriptions = [t.description for t in plan.tasks]
        if self._last_plan_task_descriptions and current_descriptions == self._last_plan_task_descriptions:
            all_completed = True
            for t in plan.tasks:
                existing = self._find_task(t.id)
                if existing is None or existing.status != "completed":
                    all_completed = False
                    break
            if all_completed and self.tasks:
                console.print("[yellow]Identical plan with no new work. Halting.[/yellow]")
                self._stop_reason = "identical plan completed"
                return

        self._upsert_tasks(plan.tasks)

        max_parallel = getattr(self.client.settings, "max_parallel_tasks", 5)
        if max_parallel < 1:
            max_parallel = 1
        sem = asyncio.Semaphore(max_parallel)

        async def run_with_sem(task: TaskModel) -> Any:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks_to_run = [run_with_sem(t) for t in plan.tasks]
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            existing = self._find_task(task.id)
            if isinstance(result, Exception):
                if existing is not None:
                    existing.status = "failed"
                    existing.result = str(result)
                else:
                    task.status = "failed"
                    task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit({"type": "task", "cycle": self.cycles, "id": str(task.id), "status": "failed", "result": str(result)})
            else:
                if existing is not None:
                    existing.status = "completed"
                    result_str = str(result)
                    existing.result = result_str
                else:
                    task.status = "completed"
                    result_str = str(result)
                    task.result = result_str
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit({"type": "task", "cycle": self.cycles, "id": str(task.id), "status": "completed", "result": str(result)})
                if result_str and result_str not in self.completed_summaries:
                    self.completed_summaries.append(result_str)

        self._last_plan_task_descriptions = current_descriptions

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self._build_goal_context(), plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.last_failures = []
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.last_failures = list(test_result.failures)
        self._emit({"type": "test", "cycle": self.cycles, "passed": bool(test_result.passed), "summary": str(test_result.summary), "failures": list(test_result.failures)})

    def _find_task(self, task_id: str) -> TaskModel | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def _upsert_tasks(self, planned_tasks: list[TaskModel]) -> None:
        for planned in planned_tasks:
            existing = self._find_task(planned.id)
            if existing is not None:
                existing.description = planned.description
                existing.files = planned.files
                existing.dependencies = planned.dependencies
                if existing.status not in ("completed", "failed"):
                    existing.status = planned.status
                if planned.result is not None and existing.result is None:
                    existing.result = planned.result
            else:
                self.tasks.append(planned.model_copy(deep=True))

    def _is_done(self) -> bool:
        if self._stop_reason is not None:
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
        return self.tasks
