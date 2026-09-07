from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import ModelState, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

EventCallback = Callable[[str, dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.settings = self.client.settings
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        self.on_event = on_event

    def _emit(self, name: str, payload: dict[str, Any] | None = None) -> None:
        if self.on_event:
            self.on_event(name, payload or {})

    async def run(self) -> None:
        state = self._load_state()
        if state is not None:
            self.goal = state.goal
            self.cycles = state.cycle
            self._tasks = state.tasks
            console.print(
                Panel.fit(
                    f"[bold green]Furrow[/bold green]\nResumed goal: {self.goal}\nCycle: {self.cycles}",
                    title="Furrow",
                )
            )
            self._emit("resumed", {"goal": self.goal, "cycle": self.cycles})
        else:
            console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit("started", {"goal": self.goal})

        while True:
            self.cycles += 1
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                self._emit("stopped", {"reason": "max_cycles", "cycle": self.cycles})
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle", {"cycle": self.cycles})
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("done", {"cycle": self.cycles})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit("plan", plan.model_dump())

        self._tasks = plan.tasks
        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [self._run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit("task_failed", {"id": task.id, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit("task_completed", {"id": task.id, "result": result})

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._last_test_result = test_result
        self._emit("test", test_result.model_dump())

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            self._emit("task_started", {"id": task.id, "description": task.description})
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._tasks
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        if completed < len(tasks):
            return False
        if self._last_test_result is not None and not self._last_test_result.passed:
            return False
        return True

    def _save_state(self) -> None:
        try:
            state = ModelState(goal=self.goal, cycle=self.cycles, tasks=self._tasks)
            Path(self.settings.state_file).write_text(state.model_dump_json(indent=2))
        except Exception:
            pass

    def _load_state(self) -> ModelState | None:
        try:
            p = Path(self.settings.state_file)
            if p.exists():
                return ModelState.model_validate_json(p.read_text())
        except Exception:
            return None
        return None
