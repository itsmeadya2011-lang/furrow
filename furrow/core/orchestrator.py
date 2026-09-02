from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

log = structlog.get_logger(__name__)

OutputCallback = Callable[[str], Awaitable[None]]

# Console for CLI output (renders rich markup).
console = Console()
# Capture console used to strip rich markup before sending to callbacks
# (e.g. WebSocket clients that don't understand rich markup).
_stripper = Console(record=True, force_terminal=False, no_color=True, width=200)


class Orchestrator:
    """Coordinates planning, parallel task execution, and testing in cycles.

    The orchestrator runs repeatedly until all tasks complete successfully, the
    configured ``max_cycles`` limit is reached, or all tasks are marked done.
    Output can be streamed to a caller-provided callback (e.g. a WebSocket).
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_output: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.on_output: OutputCallback | None = on_output

        self.cycles = 0
        self._all_tasks: list[TaskModel] = []
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    async def start(self, workspace: Path | None = None) -> None:
        """Initialise workspace context and reset per-run state."""
        if workspace is not None:
            self.settings.workspace = workspace
        self._all_tasks = []
        self.cycles = 0
        self._semaphore = asyncio.Semaphore(
            max(1, self.settings.max_parallel_tasks)
        )
        await log.ainfo(
            "orchestrator.start",
            goal=self.goal,
            max_parallel=self.settings.max_parallel_tasks,
            max_cycles=self.settings.max_cycles,
        )

    async def run(self, workspace: Path | None = None) -> None:
        """Run the main planning-execution-test loop."""
        await self.start(workspace=workspace)
        while True:
            self.cycles += 1
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                await self._emit(
                    f"\n[bright_yellow]Reached max_cycles "
                    f"({self.settings.max_cycles}). Stopping.[/bright_yellow]\n"
                )
                await log.ainfo("orchestrator.max_cycles_reached", cycles=self.cycles)
                break
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                await log.ainfo("orchestrator.done", cycles=self.cycles)
                break
            await asyncio.sleep(0)

    async def _cycle(self) -> None:
        plan = await self.planner.plan(self.goal)
        await self._emit(json.dumps(plan.model_dump(), indent=2))

        if not plan.tasks:
            await self._emit(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        self._all_tasks.extend(plan.tasks)

        workers = [
            WorkerAgent(task=task, client=self.client, workspace=self.settings.workspace)
            for task in plan.tasks
        ]
        results = await asyncio.gather(
            *(self._run_worker(w) for w in workers),
            return_exceptions=True,
        )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                await log.aerror("orchestrator.task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, worker: WorkerAgent) -> str:
        assert self._semaphore is not None
        async with self._semaphore:
            return await worker.run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        all_done = all(t.status in ("completed", "failed") for t in tasks)
        any_failed = any(t.status == "failed" for t in tasks)
        return all_done and not any_failed

    def _get_tasks(self) -> list[TaskModel]:
        return self._all_tasks

    async def _emit(self, text: str) -> None:
        """Send *text* to the output callback or the rich console.

        When ``on_output`` is set (e.g. WebSocket streaming), rich markup is
        stripped so the consumer receives plain text.  When no callback is
        registered, output is rendered through the rich console (with colours).
        """
        await log.ainfo("orchestrator.output", text=text)
        if self.on_output is not None:
            _stripper.print(text, end="", markup=True, highlight=False)
            plain = _stripper.export_text()
            _stripper.clear()
            await self.on_output(plain)
        else:
            console.print(text)
