from __future__ import annotations

import asyncio
import io
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.state import State, save_state
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


def _render_to_text(*args: Any, **kwargs: Any) -> str:
    buf = io.StringIO()
    capture = Console(
        file=buf, force_terminal=False, color_system=None, record=True
    )
    capture.print(*args, **kwargs)
    return capture.export_text()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], Awaitable[None]] | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.completed = 0
        self.failed = 0
        self.output_callback = output_callback
        self.state_path = state_path
        if state_path is not None and state_path.exists():
            saved = self._load_state_sync(state_path)
            if saved is not None:
                self.goal = saved.goal
                self.tasks = saved.tasks
                self.cycles = saved.cycles
                self.completed = saved.completed
                self.failed = saved.failed

    async def _emit(self, *args: Any, **kwargs: Any) -> None:
        console.print(*args, **kwargs)
        if self.output_callback is not None:
            await self.output_callback(_render_to_text(*args, **kwargs))

    def _update_task_counts(self) -> None:
        self.completed = sum(1 for t in self.tasks if t.status == "completed")
        self.failed = sum(1 for t in self.tasks if t.status == "failed")

    def _load_state_sync(self, path: Path) -> State | None:
        try:
            return State.model_validate_json(path.read_text())
        except (OSError, ValueError):
            return None

    async def _save_current_state(self, status: str) -> None:
        if self.state_path is None:
            return
        try:
            state = State(
                goal=self.goal,
                tasks=self.tasks,
                cycles=self.cycles,
                completed=self.completed,
                failed=self.failed,
                status=status,
            )
            await save_state(state, self.state_path)
        except (OSError, ValueError):
            return

    async def run(self) -> None:
        await self._emit(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            self._update_task_counts()
            await self._save_current_state("running")
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete")
                await self._save_current_state("completed")
                break
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit(
                    f"[yellow]Maximum cycles ({max_cycles}) reached. Halting.[/yellow]"
                )
                logger.info("max_cycles_reached", max_cycles=max_cycles)
                await self._save_current_state("max_cycles")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        logger.info("plan_created", task_count=len(plan.tasks), goal=self.goal)
        await self._emit(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks.extend(plan.tasks)
        waves = self._build_waves(plan.tasks)
        logger.info("waves_built", wave_count=len(waves))

        sem = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            for i, wave in enumerate(waves, start=1):
                logger.info("executing_wave", wave=i, task_ids=[t.id for t in wave])
                results = await asyncio.gather(
                    *[_run_task(t) for t in wave], return_exceptions=True
                )
                for task, result in zip(wave, results):
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result
                        await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result: TestResult = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )
        logger.info("test_run", passed=test_result.passed, summary=test_result.summary)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        for task in tasks:
            if task.status in ("pending", "failed"):
                return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _build_waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        done_ids: set[str] = {
            t.id for t in self.tasks if t.status in ("completed", "failed")
        }
        remaining = [t for t in tasks if t.id not in done_ids]
        waves: list[list[TaskModel]] = []
        while remaining:
            wave = [
                t for t in remaining if all(dep in done_ids for dep in t.dependencies)
            ]
            if not wave:
                for t in remaining:
                    t.status = "failed"
                    t.result = "Unresolvable dependencies"
                    logger.warning("task_unresolvable", task_id=t.id)
                break
            waves.append(wave)
            for t in wave:
                done_ids.add(t.id)
            remaining = [t for t in remaining if t.id not in done_ids]
        return waves
