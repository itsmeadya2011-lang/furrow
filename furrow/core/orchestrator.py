from __future__ import annotations

import asyncio
import json
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
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger("furrow.orchestrator")


class Orchestrator:
    """Runs the infinite planning → execution → testing loop.

    Tasks produced by the planner are tracked across cycles so that
    ``_is_done()`` reflects real progress instead of always returning
    ``True``.  ``max_cycles`` from settings bounds the loop (0 = unlimited),
    and a small JSON state file is written so progress survives across
    sessions.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self._state_path = Path(self.client.settings.workspace) / ".furrow" / "state.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self.load_state()
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        try:
            while True:
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                self.save_state()
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
                # Yield control to the event loop between cycles.
                await asyncio.sleep(0)
        finally:
            self.save_state()

    # ------------------------------------------------------------------
    # Cycle logic
    # ------------------------------------------------------------------

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(
                Pretty(self._summarize_plan(plan)),
                title="Plan",
                border_style="blue",
            )
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_result = TestResult(
                passed=True,
                summary="No tasks remaining; goal appears complete.",
                failures=[],
            )
            self.tasks = []
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result if isinstance(result, str) else str(result)
                console.print(f"[green]Task {task.id} completed[/green]")

        # The plan's task objects now carry their freshly-set statuses; make
        # them the authoritative tracked state for this cycle.
        self.tasks = plan.tasks

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await self.tester.run(self.goal, plan.tasks)

        if self.last_test_result.passed:
            console.print(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
            for failure in self.last_test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Narrow the goal to the specific failures so the next plan fixes them.
            self.goal = "Fix the following failing tests:\n" + "\n".join(
                self.last_test_result.failures
            )

    def _get_tasks(self) -> list[TaskModel]:
        """Return the tasks tracked for the current goal (used by callers/tests)."""
        return self.tasks

    # ------------------------------------------------------------------
    # State / termination
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        if max_cycles and self.cycles >= max_cycles:
            console.print("[yellow]Max cycles reached. Stopping.[/yellow]")
            log.info("max_cycles_reached", cycles=self.cycles, max=max_cycles)
            return True

        # An empty plan means the planner considers the goal complete. Only
        # treat this as done once a cycle has actually run (i.e. there is a
        # test result recorded), so a fresh, never-run orchestrator does not
        # report itself as complete.
        if not self.tasks:
            return self.last_test_result is not None

        all_completed = all(t.status == "completed" for t in self.tasks)
        any_failed = any(t.status == "failed" for t in self.tasks)

        # Tests must be green and no task may have failed to consider the goal done.
        tests_passed = self.last_test_result is not None and self.last_test_result.passed
        if tests_passed and all_completed and not any_failed:
            return True

        log.debug(
            "not_done",
            cycles=self.cycles,
            all_completed=all_completed,
            any_failed=any_failed,
            tests_passed=tests_passed,
        )
        return False

    @staticmethod
    def _summarize_plan(plan: Plan) -> dict[str, Any]:
        return {
            "rationale": plan.rationale,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "files": t.files,
                    "dependencies": t.dependencies,
                }
                for t in plan.tasks
            ],
        }

    # ------------------------------------------------------------------
    # Persistent state
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "last_test_result": self.last_test_result.model_dump()
            if self.last_test_result
            else None,
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("state_save_failed", path=str(self._state_path), error=str(exc))

    def load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
            if data.get("last_test_result"):
                self.last_test_result = TestResult(**data["last_test_result"])
        except (OSError, ValueError, TypeError) as exc:
            log.warning("state_load_failed", path=str(self._state_path), error=str(exc))
