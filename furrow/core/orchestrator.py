from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.core.state import load_state, save_state
from furrow.llm import LLMClient

logger = structlog.get_logger()
console = Console()

# Type alias for a callback that receives output lines (used by the web server
# to stream Rich console text back to WebSocket clients).
OutputHandler = Callable[[str], Awaitable[None]]

# Regex to strip Rich markup tags from strings when forwarding to plain-text handlers.
_RICH_TAG_RE = re.compile(r"\[/?[a-zA-Z0-9_# -]+\]")


def _strip_rich(text: str) -> str:
    """Remove Rich markup tags so plain-text handlers receive clean text."""
    return _RICH_TAG_RE.sub("", text)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        max_cycles: int | None = None,
        max_parallel_tasks: int | None = None,
        state_file: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0

        # --- Persistent state across cycles ---
        # All tasks from every cycle so far.  Tasks whose status is still
        # ``"pending"`` or ``"in_progress"`` from a previous cycle are carried
        # forward instead of being silently discarded.
        self.tasks: list[TaskModel] = []
        # Result of the most recent test run; used by ``_is_done``.
        self.last_test_result: TestResult | None = None

        # --- Configuration ---
        settings = self.client.settings
        # ``0`` means *unlimited* (the default in config).
        self.max_cycles = max_cycles if max_cycles is not None else settings.max_cycles
        self.max_parallel_tasks = (
            max_parallel_tasks if max_parallel_tasks is not None else settings.max_parallel_tasks
        )

        # --- Persistent state ---
        # Path to the JSON state file.  When set, the orchestrator will load
        # any saved state on init and persist after every cycle.
        if state_file is not None:
            self.state_file: Path | None = Path(state_file)
        elif resume:
            self.state_file = settings.state_file
        else:
            self.state_file = None
        self._load_persisted_state()

        # --- Output forwarding ---
        self._on_output: OutputHandler | None = None

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    def set_output_handler(self, handler: OutputHandler) -> None:
        """Register a handler that receives every line printed to the console.

        This lets the web server relay Rich output to WebSocket clients in
        real time instead of only writing to stdout.
        """
        self._on_output = handler

    async def _output(self, text: RenderableType, style: str | None = None) -> None:
        """Print to the Rich console and forward to the output handler."""
        styled: RenderableType = f"[{style}]{text}[/{style}]" if style else text
        console.print(styled)
        if self._on_output is not None:
            await self._on_output(_strip_rich(str(styled)))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal)
        await self._output(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )

        while True:
            # Enforce the cycle cap (0 == unlimited).
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self._output(
                    f"[yellow]Reached max_cycles ({self.max_cycles}). Stopping.[/yellow]",
                    style="yellow",
                )
                logger.info("orchestrator_max_cycles", cycles=self.cycles)
                break

            self.cycles += 1
            await self._output(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)

            await self._cycle()
            await self._save_persisted_state()

            if self._is_done():
                await self._output(
                    "[bold green]Goal complete. Halting.[/bold green]",
                    style="green",
                )
                logger.info("orchestrator_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        # --- Plan -------------------------------------------------------
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        await self._output(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            await self._output("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_result = None  # nothing to test
            return

        # Carry forward any previously-failed tasks so they get retried.
        # New tasks from the planner are appended; duplicate IDs are replaced.
        new_tasks = self._merge_tasks(plan.tasks)
        self.tasks.clear()
        self.tasks.extend(new_tasks)

        # Mark tasks whose dependencies are not yet done as pending.
        self._update_dependent_statuses()

        # --- Execute ----------------------------------------------------
        pending = [t for t in self.tasks if t.status == "pending"]
        # If every task is already completed/in-progress from a prior cycle
        # we don't need to re-run anything.
        if not pending and any(t.status == "in_progress" for t in self.tasks):
            pending = [t for t in self.tasks if t.status == "in_progress"]

        if pending:
            await self._output(
                f"[bold yellow]Executing {len(pending)} task(s) in parallel..."
                f" (max_parallel={self.max_parallel_tasks})[/bold yellow]"
            )
            sem = asyncio.Semaphore(self.max_parallel_tasks)
            coros = [self._run_task(task, sem) for task in pending]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(pending, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    await self._output(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    await self._output(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id)
        else:
            # All tasks were already completed in a previous cycle.
            await self._output("[dim]All tasks already complete.[/dim]")

        # --- Test -------------------------------------------------------
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            await self._output(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            await self._output(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._output(f"  • {failure}")
            await self._output("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _run_task(self, task: TaskModel, sem: asyncio.Semaphore) -> str:
        """Execute a single task respecting the global concurrency semaphore."""
        async with sem:
            task.status = "in_progress"
            return await WorkerAgent(task=task, client=self.client).run()

    # ------------------------------------------------------------------
    # Persistent state
    # ------------------------------------------------------------------
    def _load_persisted_state(self) -> None:
        """Load tasks/cycles from the state file, if one is configured.

        Only invoked when ``resume=True`` or an explicit ``state_file`` is
        provided.  If the file doesn't exist or is corrupt, we simply start
        fresh.
        """
        if self.state_file is None:
            return
        data = load_state(self.state_file)
        if data is None:
            return
        # Restore goal (but allow callers to override via the ``goal`` arg).
        if not self.goal and data.get("goal"):
            self.goal = data["goal"]
        self.cycles = data.get("cycles", 0)
        for raw in data.get("tasks", []):
            self.tasks.append(TaskModel(**raw))
        if "last_test_result" in data and data["last_test_result"] is not None:
            self.last_test_result = TestResult(**data["last_test_result"])
        logger.info("state_restored", cycles=self.cycles, tasks=len(self.tasks))

    async def _save_persisted_state(self) -> None:
        """Persist current state to the state file (if configured)."""
        if self.state_file is None:
            return
        save_state(self, self.state_file)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> list[TaskModel]:
        """Merge freshly-planned tasks with carried-over state.

        Tasks that already exist (matched by ``id``) keep their completed
        status from a previous cycle; brand-new tasks start as ``"pending"``.
        """
        existing = {t.id: t for t in self.tasks}
        merged: list[TaskModel] = []
        for task in new_tasks:
            if task.id in existing:
                # Preserve the previous status/result.
                old = existing[task.id]
                task.status = old.status
                task.result = old.result
            merged.append(task)
        return merged

    def _update_dependent_statuses(self) -> None:
        """Reset status to pending for tasks whose dependencies are still open."""
        completed_ids = {t.id for t in self.tasks if t.status == "completed"}
        for task in self.tasks:
            if task.status in ("completed",):
                continue
            if task.dependencies and not all(dep in completed_ids for dep in task.dependencies):
                task.status = "pending"

    def _get_tasks(self) -> list[TaskModel]:
        """Return all tracked tasks (used by ``_is_done`` and tests)."""
        return self.tasks

    def _is_done(self) -> bool:
        """Determine whether the orchestrator should stop.

        Stopping conditions:
        * ``max_cycles`` has been reached.
        * There are no tasks at all.
        * Every task is completed **and** the latest test run passed.
        """
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            return True

        if not self.tasks:
            return True

        # If any task failed, we need to keep going to fix it.
        if any(t.status == "failed" for t in self.tasks):
            return False

        # If any task is not yet completed, keep going.
        if any(t.status != "completed" for t in self.tasks):
            return False

        # All tasks completed — but only stop if tests also passed (or there
        # were no tests to run in this cycle).
        return self.last_test_result is None or self.last_test_result.passed
