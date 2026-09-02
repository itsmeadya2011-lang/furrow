from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TestResult
from furrow.core.state import SessionState, SessionStatus, StateManager
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the infinite Furrow development loop.

    Each cycle:
      1. Plan  – break the goal into 1-5 parallelizable tasks
      2. Execute – run worker agents in parallel
      3. Test – run the test suite and analyse results
      4. Update state and decide whether to continue

    The loop terminates when all tasks are complete and tests pass, when
    ``max_cycles`` is reached, or when no tasks are produced.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: str | Path | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.state_manager = StateManager(state_file=state_file or self.settings.workspace / ".furrow" / "state.json")
        self.original_goal = goal

        # Load existing state if present, otherwise start fresh.
        self.state_manager.load()
        if self.state_manager._state is None:
            self.state_manager.initialize(
                goal=goal,
                max_cycles=self.settings.max_cycles,
            )
        elif goal != self.state.original_goal:
            # Resume existing session; update the goal if changed.
            self.state.original_goal = goal
            self.state.goal = goal
            self.state_manager.save()

    # -- Main loop -----------------------------------------------------------

    async def run(self) -> SessionState:
        """Run the development loop until completion or cycle limit."""
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.state.original_goal}",
                title="Furrow",
            )
        )

        while True:
            if self.state_manager.is_cycle_limit_reached():
                console.print(
                    f"[yellow]Max cycles ({self.state.max_cycles}) reached. Stopping.[/yellow]"
                )
                self.state_manager.fail(f"Max cycles ({self.state.max_cycles}) reached")
                break

            self.state_manager.increment_cycle()
            console.print(f"\n[bold cyan]═══ Cycle {self.state.cycle} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                if self.state.status != SessionStatus.FAILED:
                    self.state_manager.complete()
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        self.state_manager.save()
        return self.state

    async def _cycle(self) -> None:
        """Execute a single plan → execute → test cycle."""
        # 1. Plan
        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self.state.goal)
            except ValueError as e:
                self.state_manager.add_error(str(e))
                console.print(f"[red]Planning failed: {e}[/red]")
                self.state_manager.fail(f"Planning failed: {e}")
                return

        # Record plan history
        self.state.plan_history.append(plan)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.state_manager.complete()
            return

        # 2. Execute — run all tasks in parallel
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Record task results
        self.state_manager.update_tasks(plan.tasks)
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                self.state_manager.mark_task_failed(task.id, str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self.state_manager.add_error(f"Task {task.id}: {result}")
            else:
                self.state_manager.mark_task_completed(task.id, result)
                console.print(f"[green]Task {task.id} completed[/green]")

        # 3. Test
        with Status("[bold yellow]Testing...", console=console):
            try:
                test_result = await TesterAgent(client=self.client).run(
                    self.state.goal, self.state.tasks
                )
            except Exception as e:
                test_result = TestResult(
                    passed=False,
                    summary=f"Tester agent failed: {e}",
                    failures=[str(e)],
                )

        # Record test history
        self.state.test_history.append(
            {
                "cycle": self.state.cycle,
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.state_manager.add_error(
                f"Tests failed in cycle {self.state.cycle}: {test_result.summary}"
            )
            # Update goal to focus on fixing test failures
            self.state_manager.set_goal(
                f"Fix failing tests:\n" + "\n".join(test_result.failures)
            )

    def _is_done(self) -> bool:
        """Check if the current cycle completed all tasks successfully.

        Returns True when:
        - The plan had no tasks (goal is complete), OR
        - All tasks from the plan are completed AND the last test run passed.

        Returns False when:
        - There are pending tasks, OR
        - There are failed tasks (needs retry), OR
        - Tests haven't been run yet, OR
        - The last test run failed.
        """
        tasks = self.state_manager.state.tasks
        if not tasks:
            # No tasks in the last plan — goal is complete
            return self.state.status in (SessionStatus.COMPLETED, SessionStatus.FAILED)

        # Pending tasks — work remains
        pending = [t for t in tasks if t.status == "pending"]
        if pending:
            return False

        # Failed tasks — need to retry/fix, not done
        failed = [t for t in tasks if t.status == "failed"]
        if failed:
            return False

        # All tasks completed — check if tests also passed
        if not self.state_manager.state.test_history:
            # No tests run yet
            return False

        last_test = self.state_manager.state.test_history[-1]
        return bool(last_test.get("passed", False))

    @property
    def state(self) -> SessionState:
        return self.state_manager.state
