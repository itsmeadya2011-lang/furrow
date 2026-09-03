from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, State, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Coordinates the planner → workers → tester loop until the goal is achieved.

    The orchestrator drives a fixed pipeline:
        1. Ask the ``PlannerAgent`` for a ``Plan`` of ``TaskModel``s.
        2. Execute each task in parallel via ``WorkerAgent`` (bounded by a
           semaphore sized from ``settings.max_parallel_tasks``).
        3. Run ``TesterAgent`` against the produced code.
        4. If tests fail, rewrite ``self.goal`` so the next cycle focuses on
           fixing the reported failures.
        5. Persist a ``State`` snapshot to ``settings.state_file`` so work
           survives restarts.

    Termination is decided by :py:meth:`_is_done`, which requires every task
    to have completed AND the most recent test run to have passed. Two safety
    nets stop runaway loops: ``settings.max_cycles`` (hard cap on cycles) and
    ``settings.max_consecutive_failures`` (give up if the same tests keep
    failing).
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0

        self.tasks: list[TaskModel] = []
        self.last_test_passed: bool | None = None
        self.last_failures: list[str] = []
        self._consecutive_failures = 0

        self._semaphore: asyncio.Semaphore | None = None

        self.load_state()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazily create the semaphore so it binds to the running event loop."""
        if self._semaphore is None:
            max_parallel = max(1, int(getattr(self.settings, "max_parallel_tasks", 1)))
            self._semaphore = asyncio.Semaphore(max_parallel)
        return self._semaphore

    async def run(self) -> None:
        """Run the plan/execute/test loop until done or a safety cap fires."""
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        max_cycles = int(getattr(self.settings, "max_cycles", 0))
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles={max_cycles}. Halting.[/yellow]"
                )
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        """Run one planner → workers → tester pass and persist state."""
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tasks = []
            self.save_state()
            return

        self.tasks = list(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):

            async def _run_one(task: TaskModel) -> str:
                async with self.semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            results = await asyncio.gather(
                *[_run_one(task) for task in self.tasks],
                return_exceptions=True,
            )

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
            test_result = await TesterAgent(client=self.client).run(
                self.goal, self.tasks
            )

        self.last_test_passed = test_result.passed
        self.last_failures = list(test_result.failures)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._consecutive_failures = 0
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            self._consecutive_failures += 1

            max_failures = int(
                getattr(self.settings, "max_consecutive_failures", 0)
            )
            if max_failures > 0 and self._consecutive_failures >= max_failures:
                console.print(
                    f"[red]Hit max_consecutive_failures={max_failures}. "
                    "Halting.[/red]"
                )
                self.save_state()
                raise SystemExit(1)

            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self.save_state()

    def _is_done(self) -> bool:
        """Decide whether the loop should stop.

        Returns ``True`` only when:
            - there are tasks, every task is completed, AND
            - the most recent test run passed.
        Returns ``False`` if any task failed, or if tests are still failing
        (so the next cycle has a chance to retry).
        """
        if not self.tasks:
            return self.last_test_passed is not False

        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")

        if failed > 0:
            return False
        if completed < len(self.tasks):
            return False
        if self.last_test_passed is not True:
            return False
        return True

    def save_state(self) -> None:
        """Write the current orchestrator state to ``settings.state_file``."""
        state_path = Path(self.settings.state_file)
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = State(
                goal=self.goal,
                cycles=self.cycles,
                tasks=self.tasks,
                last_test_passed=self.last_test_passed,
                last_failures=self.last_failures,
                consecutive_failures=self._consecutive_failures,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            state_path.write_text(json.dumps(snapshot.model_dump(), indent=2))
        except OSError as exc:
            console.print(f"[yellow]Could not save state: {exc}[/yellow]")

    def load_state(self) -> None:
        """Restore orchestrator state from ``settings.state_file`` if present."""
        state_path = Path(self.settings.state_file)
        if not state_path.exists():
            return
        try:
            raw = json.loads(state_path.read_text())
            snapshot = State.model_validate(raw)
        except (OSError, ValueError) as exc:
            console.print(f"[yellow]Could not load state: {exc}[/yellow]")
            return

        self.goal = snapshot.goal or self.goal
        self.cycles = int(snapshot.cycles)
        self.tasks = list(snapshot.tasks)
        self.last_test_passed = snapshot.last_test_passed
        self.last_failures = list(snapshot.last_failures)
        self._consecutive_failures = int(snapshot.consecutive_failures)