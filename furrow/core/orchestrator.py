from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, TaskModel
from furrow.llm import LLMClient

log = structlog.get_logger(__name__)


class OrchestratorState:
    """Tracks cycle history, failed tasks, and goal modifications."""

    def __init__(self) -> None:
        self.cycles: list[dict[str, Any]] = []
        self.failed_tasks: list[TaskModel] = []
        self.goal_history: list[str] = []

    def record_cycle(
        self,
        cycle_number: int,
        plan: Plan,
        test_result: TestResult | None,
    ) -> None:
        self.cycles.append(
            {
                "cycle": cycle_number,
                "plan": plan.model_dump(),
                "test_result": test_result.model_dump() if test_result else None,
            }
        )

    def add_failed_task(self, task: TaskModel) -> None:
        self.failed_tasks.append(task)

    def record_goal(self, goal: str) -> None:
        self.goal_history.append(goal)


class Orchestrator:
    """Autonomous coding orchestrator that runs planning → execution → testing cycles."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.goal = goal
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.console = console or Console()
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.last_cycle_tasks: list[TaskModel] = []
        self.state = OrchestratorState()
        self.state.record_goal(goal)

    async def run(self) -> None:
        log.info("orchestrator.start", goal=self.goal)
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            log.info("cycle.start", cycle=self.cycles)
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                log.info("goal.complete", cycles=self.cycles)
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            # Guard against infinite loops when planner consistently returns no tasks
            if not self.last_cycle_tasks:
                log.warning("orchestrator.no_tasks_persistent", cycle=self.cycles)
                self.console.print("[yellow]No tasks produced for 3 consecutive cycles. Stopping.[/yellow]")
                break

            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                log.info("max_cycles.reached", max_cycles=self.settings.max_cycles)
                self.console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self._plan_with_retry(self.goal)
        log.info("plan.received", task_count=len(plan.tasks), rationale=plan.rationale)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            log.warning("plan.no_tasks", goal=self.goal)
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute_tasks(plan)
        self.last_cycle_tasks = [t for t in plan.tasks]

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await self._test_with_retry(self.goal, plan.tasks)
        self.last_test_result = test_result
        self.state.record_cycle(self.cycles, plan, test_result)

        if test_result.passed:
            log.info("tests.passed", summary=test_result.summary)
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            log.warning("tests.failed", summary=test_result.summary, failures=test_result.failures)
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  - {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._update_goal_for_failures(test_result)

    async def _plan_with_retry(self, goal: str) -> Plan:
        """Call the planner with retry logic on transient failures."""
        retrying = AsyncRetrying(
            retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        )
        context = self._build_plan_context()
        async for attempt in retrying:
            with attempt:
                return await self.planner.plan(goal, context=context)
        # Unreachable: AsyncRetrying always yields
        raise RuntimeError("plan retry exhausted")  # pragma: no cover

    def _build_plan_context(self) -> dict[str, Any] | None:
        """Build context from previous cycle failures and test results."""
        failed_tasks = [t.description for t in self.state.failed_tasks]
        test_results: list[str] = []
        if self.last_test_result is not None and not self.last_test_result.passed:
            test_results.append(self.last_test_result.summary)
            test_results.extend(self.last_test_result.failures)
        if not failed_tasks and not test_results:
            return None
        context: dict[str, Any] = {}
        if failed_tasks:
            context["failed_tasks"] = failed_tasks
        if test_results:
            context["test_results"] = test_results
        return context

    async def _execute_tasks(self, plan: Plan) -> None:
        """Execute planned tasks, batching when count exceeds max_parallel_tasks."""
        tasks = plan.tasks
        max_parallel = self.settings.max_parallel_tasks

        if max_parallel > 0 and len(tasks) > max_parallel:
            log.info("tasks.batching", total=len(tasks), batch_size=max_parallel)
            self.console.print(f"[dim]Batching {len(tasks)} tasks into groups of {max_parallel}[/dim]")
            for i in range(0, len(tasks), max_parallel):
                batch = tasks[i : i + max_parallel]
                await self._run_batch(batch)
        else:
            await self._run_batch(tasks)

    async def _run_batch(self, batch: list[TaskModel]) -> None:
        """Run a batch of worker tasks concurrently and record results."""
        with Status("[bold yellow]Executing tasks...", console=self.console):
            coros = [WorkerAgent(task=task, client=self.client, settings=self.settings).run() for task in batch]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(batch, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.state.add_failed_task(task)
                log.error("task.failed", task_id=task.id, error=str(result))
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                log.info("task.completed", task_id=task.id)
                self.console.print(f"[green]Task {task.id} completed[/green]")
            self.all_tasks.append(task)

    async def _test_with_retry(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Call the tester with retry logic on transient failures."""
        retrying = AsyncRetrying(
            retry=retry_if_exception_type((Exception,)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        )
        async for attempt in retrying:
            with attempt:
                return await TesterAgent(client=self.client, settings=self.settings).run(goal, tasks)
        # Unreachable: AsyncRetrying always yields
        raise RuntimeError("test retry exhausted")  # pragma: no cover

    def _update_goal_for_failures(self, test_result: TestResult) -> None:
        """Update the goal to focus on failures while preserving original context."""
        if not test_result.failures:
            return
        original = self.state.goal_history[0] if self.state.goal_history else self.goal
        self.goal = (
            f"Original goal: {original}\n\n"
            f"Fix the following failing tests:\n" + "\n".join(test_result.failures)
        )
        self.state.record_goal(self.goal)

    def _get_tasks(self) -> list[TaskModel]:
        """Return all accumulated tasks across cycles."""
        return self.all_tasks

    def _is_done(self) -> bool:
        """Check if the most recent cycle's tasks are all completed and tests pass."""
        if not self.last_cycle_tasks:
            return False
        completed = sum(1 for t in self.last_cycle_tasks if t.status == "completed")
        failed = sum(1 for t in self.last_cycle_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed < len(self.last_cycle_tasks):
            return False
        if self.last_test_result is not None and not self.last_test_result.passed:
            return False
        return True