from __future__ import annotations

import asyncio
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
from furrow.config import AgentState, CycleState, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._history: list[CycleState] = []
        self._last_test_result: TestResult | None = None
        self._override_max_cycles: int | None = None
        self._status_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    @property
    def max_cycles(self) -> int:
        return self.client.settings.max_cycles

    @property
    def state_file(self) -> Path:
        workspace = getattr(self.client.settings, "workspace", None)
        base = Path(workspace) if workspace is not None else Path.cwd()
        return base / ".furrow" / "state.json"

    async def run(
        self,
        max_cycles: int | None = None,
        status_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._override_max_cycles = max_cycles if max_cycles is not None and max_cycles > 0 else None
        self._status_callback = status_callback
        effective_max_cycles = self._override_max_cycles or self.max_cycles
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if self._status_callback is not None:
            await self._status_callback({"type": "start", "goal": self.goal, "cycles": self.cycles})
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            should_continue = await self._cycle(max_cycles=max_cycles, status_callback=status_callback)
            if not should_continue:
                break
            if effective_max_cycles > 0 and self.cycles >= effective_max_cycles:
                console.print("[yellow]Max cycles reached. Stopping.[/yellow]")
                break
            if self._is_done() and (self._last_test_result is None or self._last_test_result.passed):
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
        if self._status_callback is not None:
            await self._status_callback({"type": "complete", "goal": self.goal, "cycles": self.cycles})

    async def _cycle(
        self,
        max_cycles: int | None = None,
        status_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> bool:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.tasks = list(plan.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if status_callback is not None:
            await status_callback({"type": "planning_done", "tasks": len(self.tasks), "cycles": self.cycles})

        if not plan.tasks:
            console.print(
                "[yellow]No tasks planned. Goal may be complete or no further action possible.[/yellow]"
            )
            self.save_state()
            return False

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_worker(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[_run_worker(task) for task in plan.tasks], return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            if status_callback is not None:
                await status_callback({
                    "type": "task_done",
                    "task_id": task.id,
                    "status": "completed" if not isinstance(result, Exception) else "failed",
                    "cycles": self.cycles,
                })

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._last_test_result = test_result
        if status_callback is not None:
            await status_callback({
                "type": "test_done",
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
                "cycles": self.cycles,
            })

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self._history.append(
            CycleState(cycle_number=self.cycles, tasks=list(self.tasks), test_result=test_result)
        )
        self.save_state()
        return True

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        return failed == 0 and completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    def save_state(self) -> None:
        try:
            state = AgentState(
                goal=self.goal,
                cycles=self.cycles,
                current_cycle=self._history[-1] if self._history else None,
                history=self._history[:-1],
            )
            state_file = self.state_file
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(state.model_dump_json(indent=2))
        except (OSError, ValueError, TypeError, AttributeError) as e:
            console.print(f"[dim red]Failed to save state: {e}[/dim red]")

    def load_state(self) -> None:
        try:
            state_file = self.state_file
            if not state_file.exists():
                return
            state = AgentState.model_validate_json(state_file.read_text())
        except (OSError, ValueError, TypeError):
            return
        self.goal = state.goal
        self.cycles = state.cycles
        if state.current_cycle is not None:
            self.tasks = list(state.current_cycle.tasks)
            self._last_test_result = state.current_cycle.test_result
        self._history = list(state.history)
        if state.current_cycle is not None:
            self._history.append(state.current_cycle)
