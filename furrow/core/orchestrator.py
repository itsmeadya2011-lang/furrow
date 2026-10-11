from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import OrchestratorState, Settings, TaskModel, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("orchestrator")


class TaskExecutionError(Exception):
    """Raised when all retry attempts for a task fail."""


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings = settings,
        resume: bool = False,
    ) -> None:
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client, settings=settings)
        self.state_file = settings.state_file
        self.state: OrchestratorState | None = None
        self.goal = goal
        self.resume = resume
        self._context_tokens = 0

    async def run(self) -> None:
        if self.resume:
            await self._load_state()
        else:
            self.state = OrchestratorState(goal=self.goal)
            await self._save_state()

        state = self.state
        if state is None:
            state = OrchestratorState(goal=self.goal)
            self.state = state
            await self._save_state()

        logger.info("furrow_started", goal=state.goal)
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {state.goal}", title="Furrow")
        )

        while True:
            state.cycles += 1
            logger.info("cycle_started", cycle=state.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {state.cycles} ═══[/bold cyan]")

            # Check max_cycles limit
            if self.settings.max_cycles > 0 and state.cycles > self.settings.max_cycles:
                logger.info("max_cycles_reached", max_cycles=self.settings.max_cycles)
                console.print(f"[yellow]Reached max cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                break

            await self._cycle()

            # Compact context if needed
            await self._maybe_compact_context()

            if self._is_done():
                logger.info("goal_completed", cycles=state.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._cleanup_state()
                break

    async def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                content = self.state_file.read_text()
                self.state = OrchestratorState.model_validate_json(content)
                logger.info("state_loaded", cycles=self.state.cycles, tasks_count=len(self.state.tasks))
                console.print(f"[green]Resumed from state file: {self.state.cycles} cycles completed[/green]")
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.error("state_load_failed", error=str(e))
                console.print(f"[yellow]Failed to load state: {e}. Starting fresh.[/yellow]")
                self.state = OrchestratorState(goal=self.goal)
        else:
            logger.info("no_state_file")
            console.print("[yellow]No state file found. Starting fresh.[/yellow]")
            self.state = OrchestratorState(goal=self.goal)

    async def _save_state(self) -> None:
        if self.state:
            self.state.updated_at = datetime.now()
            self.state_file.write_text(self.state.model_dump_json(indent=2))
            logger.debug("state_saved", cycles=self.state.cycles)

    async def _cleanup_state(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("state_cleaned_up")
            console.print("[green]State file cleaned up.[/green]")

    async def _maybe_compact_context(self) -> None:
        """Compact context if it's getting too large."""
        if not self.state:
            return
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
        total_chars = len(self.state.goal)
        for task in self.state.tasks:
            total_chars += len(task.description) + len(task.result or "")
        if self.state.last_test_result:
            total_chars += len(self.state.last_test_result.summary)
            total_chars += sum(len(f) for f in self.state.last_test_result.failures)
        
        estimated_tokens = total_chars // 4
        
        # If over 50k tokens, compact
        if estimated_tokens > 50000:
            logger.info("compacting_context", estimated_tokens=estimated_tokens)
            await self._compact_context()

    async def _compact_context(self) -> None:
        """Compact the state by summarizing completed tasks."""
        if not self.state:
            return

        # Keep only the most recent 10 completed tasks with full details
        # Summarize older ones
        completed = [t for t in self.state.tasks if t.status == "completed"]
        if len(completed) > 10:
            # Summarize older completed tasks
            for task in completed[:-10]:
                if task.result and len(task.result) > 200:
                    task.result = task.result[:200] + "... [compacted]"

        # Compact last test result if large
        if (
            self.state.last_test_result
            and self.state.last_test_result.summary
            and len(self.state.last_test_result.summary) > 500
        ):
            self.state.last_test_result.summary = (
                self.state.last_test_result.summary[:500] + "... [compacted]"
            )

        await self._save_state()
        logger.info("context_compacted")

    async def _cycle(self) -> None:
        state = self.state
        if state is None:
            return
        # Plan new tasks based on current state
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(state.goal, state)

        logger.info("plan_created", tasks_count=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to state
        for task in plan.tasks:
            task.cycle = state.cycles
            state.tasks.append(task)

        await self._save_state()

        # Execute tasks respecting dependencies
        await self._execute_tasks(plan.tasks)

        # Test
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                state.goal, state.tasks
            )

        state.last_test_result = test_result

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal to focus on fixing tests
            state.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        await self._save_state()

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        # Build dependency graph
        completed_ids: set[str] = set()
        max_retries = 3

        while True:
            # Find ready tasks (all dependencies completed, not yet running/completed)
            ready = [
                task
                for task in tasks
                if task.status == "pending"
                and all(dep in completed_ids for dep in task.dependencies)
            ]

            if not ready:
                # Check if any pending tasks remain
                pending = [t for t in tasks if t.status == "pending"]
                if not pending:
                    break
                # Check for circular dependencies or all blocked
                running = [t for t in tasks if t.status == "running"]
                if not running:
                    logger.error("tasks_blocked", pending_ids=[t.id for t in pending])
                    console.print("[red]All pending tasks blocked by unmet dependencies[/red]")
                    for task in pending:
                        task.status = "failed"
                        task.result = "Blocked by unmet dependencies"
                    break
                # Wait for running tasks to complete
                await asyncio.sleep(0.5)
                continue

            # Limit parallel execution
            batch = ready[: self.settings.max_parallel_tasks]
            for task in batch:
                task.status = "running"

            logger.info("executing_batch", task_ids=[t.id for t in batch])
            console.print(f"[cyan]Executing {len(batch)} tasks in parallel...[/cyan]")

            # Run batch
            results = await asyncio.gather(
                *[self._run_task_with_retry(task, max_retries) for task in batch],
                return_exceptions=True,
            )

            for task, result in zip(batch, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    task.attempts += 1
                    logger.error("task_failed", task_id=task.id, error=str(result))
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = str(result)
                    completed_ids.add(task.id)
                    logger.info("task_completed", task_id=task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")

            await self._save_state()

    async def _run_task_with_retry(self, task: TaskModel, max_retries: int) -> str:
        last_error: BaseException | None = None
        for attempt in range(max_retries):
            task.attempts = attempt + 1
            logger.debug("task_attempt", task_id=task.id, attempt=attempt + 1)
            try:
                result = await WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                return result
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning("task_attempt_failed", task_id=task.id, attempt=attempt + 1, error=str(e))
                console.print(f"[yellow]Task {task.id} attempt {attempt + 1} failed: {e}[/yellow]")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
        if last_error is not None:
            raise last_error
        raise TaskExecutionError("Unknown error")

    def _is_done(self) -> bool:
        if not self.state:
            return False
        all_tasks = self.state.tasks + self.state.completed_tasks + self.state.failed_tasks
        if not all_tasks:
            return False
        completed = sum(1 for t in all_tasks if t.status == "completed")
        failed = sum(1 for t in all_tasks if t.status == "failed")
        pending = sum(1 for t in all_tasks if t.status == "pending")
        running = sum(1 for t in all_tasks if t.status == "running")
        # Done if no pending/running tasks and all completed (or some failed but we can't continue)
        if pending == 0 and running == 0:
            if failed > 0:
                console.print(f"[red]{failed} tasks failed. Cannot continue.[/red]")
            return completed >= len(all_tasks) - failed
        return False

    def _get_tasks(self) -> list[Any]:
        if not self.state:
            return []
        return self.state.tasks + self.state.completed_tasks + self.state.failed_tasks
