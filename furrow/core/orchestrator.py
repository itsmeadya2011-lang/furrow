from __future__ import annotations

import asyncio

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.core.state import AgentState, CycleRecord
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)
console = Console()


class Orchestrator:
    """Runs the infinite development loop.

    The loop is: plan → execute (parallel workers) → test → report → repeat.

    Configurable via ``furrow.config.Settings``:
      - ``max_cycles``: hard limit on cycles (0 = unlimited)
      - ``max_parallel_tasks``: semaphore to limit concurrent workers
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state: AgentState | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None

        # Load or create persistent state
        if state is not None:
            self.state = state
        else:
            self.state = AgentState.load(self.client.settings.workspace)
        self.state.goal = self.goal
        self.state.running = True

        # Concurrency limiter for parallel workers
        max_parallel = self.client.settings.max_parallel_tasks or 5
        self._semaphore = asyncio.Semaphore(max_parallel)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        logger.info("orchestrator started", goal=self.goal)

        try:
            while True:
                self.cycles += 1
                logger.info("starting cycle", cycle=self.cycles)
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

                # Resume any previously failed tasks first
                if self.cycles > 1 and self.state.failed_tasks:
                    console.print(
                        f"[yellow]Resuming {len(self.state.failed_tasks)} "
                        f"failed task(s) from previous cycle.[/yellow]"
                    )

                await self._cycle()

                # Persist state after each cycle
                self.state.save(self.client.settings.workspace)

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    logger.info("goal complete", cycles=self.cycles)
                    self.state.running = False
                    break

                # Check max_cycles limit
                if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                    console.print(
                        f"[yellow]Reached max_cycles limit "
                        f"({self.client.settings.max_cycles}). Halting.[/yellow]"
                    )
                    logger.info("max cycles reached", max_cycles=self.client.settings.max_cycles)
                    self.state.running = False
                    break
        finally:
            self.state.running = False
            self.state.save(self.client.settings.workspace)

    async def _cycle(self) -> None:
        # ---- 1. Plan ----
        with Status("[bold yellow]Planning...", console=console) as _:
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan created", tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            # Record an empty cycle for completeness
            self.state.add_cycle(
                CycleRecord(cycle=self.cycles, plan=plan.model_dump(), tasks=[])
            )
            return

        # ---- 2. Execute (parallel, bounded by semaphore) ----
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                asyncio.create_task(
                    self._run_worker_bounded(task)
                )
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_tasks: list[TaskModel] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task completed", task_id=task.id)
            cycle_tasks.append(task)

        # ---- 3. Test ----
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        test_result_dict = test_result.model_dump()

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            logger.warning("tests failed", failures=test_result.failures)
            # Focus next planning cycle on fixing failures
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        # ---- 4. Record cycle in state ----
        self.state.add_cycle(
            CycleRecord(
                cycle=self.cycles,
                plan=plan.model_dump(),
                tasks=cycle_tasks,
                test_result=test_result_dict,
            )
        )

    async def _run_worker_bounded(self, task: TaskModel) -> str:
        """Run a worker with concurrency limiting via semaphore."""
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    # ------------------------------------------------------------------ #
    # Completion check
    # ------------------------------------------------------------------ #

    def _is_done(self) -> bool:
        """Check whether the goal is fully complete.

        The goal is done when there are no pending or failed tasks.
        """
        # If we have state from previous cycles, use it
        pending = self.state.pending_tasks
        if pending:
            return False
        # If no tasks were ever created (empty cycles), check if plan was empty
        all_tasks = self.state.all_tasks
        if not all_tasks:
            # No tasks recorded — only done if planner returned no tasks
            return self.current_plan is not None and len(self.current_plan.tasks) == 0
        # All tasks completed (no pending, no failed)
        return True

    def _get_tasks(self) -> list[TaskModel]:
        """Return all tasks tracked across cycles."""
        return self.state.all_tasks
