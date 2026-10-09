from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger(__name__)

# Type alias for the log callback used by the web UI
LogCallback = Callable[[str], Awaitable[None]] | None


class Orchestrator:
    """Runs the Furrow infinite development loop.

    Cycle: Plan -> Execute (dependency-aware parallel) -> Test -> Self-heal on failure.
    Tracks task state across cycles and enforces ``max_cycles`` (0 = unlimited).

    Parameters
    ----------
    goal:
        The high-level objective to pursue.
    client:
        Optional LLMClient instance (uses default settings if omitted).
    settings:
        Optional Settings override.
    state_file:
        Path to the state file for persistence.  Defaults to
        ``<workspace>/.furrow-state.json``.
    log_callback:
        Optional async callback invoked with log messages.  Used by the
        web UI to stream progress back to the browser via WebSocket.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: str | Path | None = None,
        log_callback: LogCallback = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else Settings())
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._log_callback = log_callback

        # State persistence
        self.state_file = Path(state_file) if state_file else self._default_state_file()
        self._load_state()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    async def _emit(self, message: str) -> None:
        """Send a log line to both the console and an optional callback."""
        console.print(message)
        if self._log_callback is not None:
            await self._log_callback(message)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _default_state_file(self) -> Path:
        return self.settings.workspace / ".furrow-state.json"

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.tasks = [
                    TaskModel(**t) for t in data.get("tasks", [])
                ]
                log.info("Loaded saved state", goal=self.goal, cycles=self.cycles, tasks=len(self.tasks))
            except Exception as e:
                log.warning("Failed to load state file", error=str(e))

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            log.debug("State saved", file=str(self.state_file))
        except Exception as e:
            log.warning("Failed to save state file", error=str(e))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self._emit(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )

        max_cycles = self.settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit(
                    f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]"
                )
                log.info("Max cycles reached, stopping", max_cycles=max_cycles)
                break

            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                log.info("Goal complete, halting")
                self._cleanup_state()
                break

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    async def _cycle(self) -> None:
        plan = await self.planner.plan(self.goal, prior_tasks=self.tasks)
        await self._emit(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        log.debug("Plan created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            log.info("No tasks in plan")
            return

        # Merge new tasks with existing tracked tasks
        self.tasks.extend(plan.tasks)

        # Execute tasks respecting dependencies (topological order)
        await self._execute_with_deps(plan.tasks)

        # Test
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        log.debug("Test result", passed=test_result.passed, summary=test_result.summary)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  [red]• {failure}[/red]")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_with_deps(self, tasks: list[TaskModel]) -> list[TaskModel | Exception]:
        """Execute tasks respecting their dependency graph.

        Tasks whose dependencies have not completed yet are deferred to
        the next scheduling wave. Within a wave, independent tasks run in
        parallel via ``asyncio.gather``.
        """
        results: list[TaskModel | Exception] = []
        pending = list(tasks)
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()

        while pending:
            # Tasks whose dependencies are all satisfied this round
            ready = [
                t for t in pending
                if all(
                    dep in completed_ids or dep in failed_ids
                    for dep in t.dependencies
                )
            ]

            if not ready:
                # Circular or unresolvable dependency -- execute remaining as best effort
                log.warning("Circular/unresolvable deps, executing remaining tasks", ready=len(pending))
                ready = pending[:]

            coros = [WorkerAgent(task=t, client=self.client).run() for t in ready]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    failed_ids.add(task.id)
                    await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                    log.error("Task failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    completed_ids.add(task.id)
                    await self._emit(f"[green]Task {task.id} completed[/green]")
                    log.debug("Task completed", task_id=task.id)
                results.append(task)
                pending.remove(task)

        return results

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        """Return True when all tracked tasks are completed and none failed."""
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def _get_tasks(self) -> list[TaskModel]:
        """Return the current tracked tasks (kept for API compatibility)."""
        return self.tasks

    def _cleanup_state(self) -> None:
        """Remove the state file once the goal is fully complete."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                log.debug("Cleaned up state file", file=str(self.state_file))
        except Exception as e:
            log.warning("Failed to clean up state file", error=str(e))
