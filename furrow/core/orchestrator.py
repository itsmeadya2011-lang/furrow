from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

CompletionCallback = Callable[[str, object], Awaitable[None]] | None


class Orchestrator:
    """Runs the infinite planning → executing → testing development loop.

    Parameters
    ----------
    goal:
        The high-level objective to achieve.
    client:
        Optional pre-configured :class:`LLMClient`.  If omitted a new client is
        created from the singleton :data:`~furrow.config.settings`.
    settings:
        Optional :class:`Settings` instance used for ``max_cycles`` enforcement
        and model routing.  Defaults to the module-level ``settings``.
    on_event:
        Optional async callback ``(event_type: str, payload) -> None`` invoked
        at key points in the loop (used by the web UI to stream progress).
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: CompletionCallback = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.current_plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self._on_event = on_event
        self._done = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Run the development loop until the goal is complete, an error is
        unrecoverable, or ``max_cycles`` is reached."""
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )

        while not self._done:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            cycle_ok = await self._cycle()
            if not cycle_ok:
                # Unrecoverable error occurred in the cycle.
                break

            self._done = self._is_done()
            if self._done:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("complete", {"cycle": self.cycles})
                break

    # ------------------------------------------------------------------ #
    # Core cycle
    # ------------------------------------------------------------------ #

    async def _cycle(self) -> bool:
        """Execute a single planning → execution → testing cycle.

        Returns ``False`` if the cycle could not complete (e.g. the planner
        returned no parseable plan), signalling the caller to stop the loop.
        """

        await self._emit("cycle_start", {"cycle": self.cycles})

        # 1. Plan ---------------------------------------------------------- #
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.goal)
        except ValueError as exc:
            console.print(f"[red]Planning failed: {exc}[/red]")
            await self._emit("error", {"stage": "plan", "error": str(exc)})
            return False

        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan", plan.model_dump())

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("info", {"message": "No tasks planned"})
            self._done = True
            return True

        # Accumulate tasks for cross-cycle state tracking.
        self.all_tasks.extend(plan.tasks)

        # 2. Execute ------------------------------------------------------- #
        await self._execute_tasks(plan)

        # 3. Test ---------------------------------------------------------- #
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result
        await self._emit("test", test_result.model_dump())

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit("test_passed", {"summary": test_result.summary})
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._emit(
                "test_failed",
                {"summary": test_result.summary, "failures": test_result.failures},
            )
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return True

    async def _execute_tasks(self, plan: Plan) -> None:
        """Run all tasks in ``plan`` in parallel, respecting ``max_parallel_tasks``."""
        max_parallel = self.settings.max_parallel_tasks
        tasks = plan.tasks
        console.print(f"[bold]Executing {len(tasks)} tasks (max parallel: {max_parallel})[/bold]")
        await self._emit("execution_start", {"task_count": len(tasks)})

        # Process tasks in batches of ``max_parallel`` to avoid overwhelming
        # the LLM API with too many concurrent requests.
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i : i + max_parallel]
            coros = [
                WorkerAgent(task=task, client=self.client).run() for task in batch
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    await self._emit(
                        "task_failed",
                        {"task_id": task.id, "error": str(result)},
                    )
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    await self._emit(
                        "task_completed",
                        {"task_id": task.id, "result": result},
                    )

    # ------------------------------------------------------------------ #
    # Completion logic
    # ------------------------------------------------------------------ #

    def _is_done(self) -> bool:
        """Determine whether the development loop should stop.

        Completion requires *all* of the following:
        1. Every task in the most recent plan completed successfully.
        2. The tests for that plan passed.
        3. The ``max_cycles`` limit (if set) has not been exceeded.
        """

        # Respect max_cycles (0 = unlimited).
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            console.print(
                f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
            )
            return True

        plan = self.current_plan
        if plan is None or not plan.tasks:
            # No plan was produced — nothing to do this cycle.
            return True

        all_completed = all(t.status == "completed" for t in plan.tasks)
        tests_passed = self.last_test_result is not None and self.last_test_result.passed

        return bool(all_completed and tests_passed)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _emit(self, event_type: str, payload: object) -> None:
        """Fire an event to the registered callback (if any)."""
        if self._on_event is not None:
            try:
                await self._on_event(event_type, payload)
            except Exception:  # noqa: BLE001, S110
                # Never let event callback errors break the development loop.
                pass
