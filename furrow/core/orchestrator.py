from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from structlog import get_logger

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

logger = get_logger(__name__)


class Orchestrator:
    """Coordinates the infinite planning → execution → testing loop."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
        settings: Settings = default_settings,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.console = console or Console()
        self.settings = settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self._workspace_cache: list[str] | None = None

    async def run(self) -> None:
        self.console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as exc:  # noqa: BLE001
                logger.error("Cycle failed", cycle=self.cycles, error=str(exc))
                self.console.print(
                    f"[red]Cycle {self.cycles} failed: {exc}[/red]"
                )
                for t in self._get_tasks():
                    if t.status == "pending":
                        t.status = "failed"
                if self._is_done():
                    self.console.print(
                        "[bold green]Goal complete. Halting.[/bold green]"
                    )
                    break
                self.console.print(
                    "[yellow]Will retry in next cycle.[/yellow]"
                )
                continue

            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        workspace_files = await self._gather_workspace_context()
        context = self._build_planning_context(workspace_files)

        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal, context=context)
        self.console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            self.console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        # Merge new tasks into all_tasks for tracking across cycles
        self.all_tasks.extend(plan.tasks)

        # Execute tasks in dependency-aware batches
        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        batches = self._topological_batches(plan.tasks)
        for batch in batches:
            results = await asyncio.gather(
                *[_run_task(task) for task in batch], return_exceptions=True
            )
            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self.console.print(
                        f"[red]Task {task.id} failed: {result}[/red]"
                    )
                else:
                    task.status = "completed"
                    task.result = result
                    self.console.print(
                        f"[green]Task {task.id} completed[/green]"
                    )

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            self.console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
        else:
            self.console.print(
                f"[red]Tests failed: {test_result.summary}[/red]"
            )
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print(
                "[yellow]Will attempt fix in next cycle.[/yellow]"
            )
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _gather_workspace_context(self) -> list[str]:
        """Gather workspace file listing (cached per orchestrator instance)."""
        if self._workspace_cache is None:
            self._workspace_cache = await self.client.list_files(
                self.settings.workspace
            )
        return self._workspace_cache

    def _build_planning_context(self, workspace_files: list[str]) -> str:
        """Build a context string for the planner."""
        if not workspace_files:
            return "Workspace is empty."
        lines = ["Existing workspace files:"]
        for f in workspace_files[:50]:
            lines.append(f"  - {f}")
        if len(workspace_files) > 50:
            lines.append(f"  ... and {len(workspace_files) - 50} more files.")
        return "\n".join(lines)

    @staticmethod
    def _topological_batches(tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group tasks into batches respecting dependencies.

        Tasks in the same batch are independent and can run in parallel.
        Tasks in batch N+1 depend on completion of tasks in earlier batches.
        """
        by_id = {t.id: t for t in tasks}
        remaining = set(by_id.keys())
        batches: list[list[TaskModel]] = []

        while remaining:
            # Find tasks whose deps are all satisfied
            ready = [
                by_id[tid]
                for tid in remaining
                if not by_id[tid].dependencies
                or all(dep not in remaining for dep in by_id[tid].dependencies)
            ]
            if not ready:
                # Circular dependency fallback: just take all remaining
                ready = [by_id[tid] for tid in remaining]
            ready_ids = {t.id for t in ready}
            remaining -= ready_ids
            batches.append(ready)

        return batches

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
