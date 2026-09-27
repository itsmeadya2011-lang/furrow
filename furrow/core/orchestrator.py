from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.core.state import FurrowState
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)
console = Console()


class Orchestrator:
    MAX_WORKER_RETRIES = 2

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state: FurrowState | None = None,
        enable_state: bool = True,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan_tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None

        self.enable_state = enable_state
        if enable_state:
            self.state = state or FurrowState.load() or FurrowState()
            if self.state.goal and self.state.status == "running":
                logger.info("resuming_from_previous_state", cycles=self.state.cycles)
                self.cycles = self.state.cycles
                self.goal = self.state.goal
        else:
            self.state = None

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            )
        )
        logger.info("orchestrator_start", goal=self.goal)
        if self.enable_state:
            self.state.set_goal(self.goal)
            self.state.set_status("running")
            self.state.save()
        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycle=self.cycles)
                if self.enable_state:
                    self.state.cycles = self.cycles
                    self.state.set_status("completed")
                    self.state.save()
                break
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]"
                )
                logger.info("max_cycles_reached", cycle=self.cycles, max_cycles=max_cycles)
                if self.enable_state:
                    self.state.cycles = self.cycles
                    self.state.set_status("failed")
                    self.state.save()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        logger.info("plan_generated", tasks=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        self.current_plan_tasks = plan.tasks
        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_with_retry(task: TaskModel) -> None:
            async with semaphore:
                for attempt in range(self.MAX_WORKER_RETRIES + 1):
                    try:
                        result = await WorkerAgent(task=task, client=self.client).run()
                        task.status = "completed"
                        task.result = result
                        logger.info("task_completed", task_id=task.id, attempts=attempt + 1)
                        return
                    except Exception as e:
                        logger.warning(
                            "task_attempt_failed",
                            task_id=task.id,
                            attempt=attempt + 1,
                            error=str(e),
                        )
                        if attempt < self.MAX_WORKER_RETRIES:
                            task.description = (
                                f"{task.description}\n\n"
                                f"Fix instructions: The previous attempt failed with "
                                f"error: {e}. Fix the issue and complete the task."
                            )
                            continue
                        task.status = "failed"
                        task.result = str(e)
                        logger.error("task_failed", task_id=task.id, error=str(e))

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            await asyncio.gather(*(run_with_retry(task) for task in plan.tasks))

        for task in plan.tasks:
            if task.status == "completed":
                console.print(f"[green]Task {task.id} completed[/green]")
            else:
                console.print(f"[red]Task {task.id} failed: {task.result}[/red]")

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if self.last_test_result.passed:
            console.print(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
            logger.info("tests_passed", summary=self.last_test_result.summary)
        else:
            console.print(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
            for failure in self.last_test_result.failures:
                console.print(f"  • {failure}")
            logger.warning("tests_failed", summary=self.last_test_result.summary)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(
                self.last_test_result.failures
            )

        if self.enable_state:
            self.state.cycles = self.cycles
            self.state.set_goal(self.goal)
            self.state.tasks = list(self.current_plan_tasks)
            self.state.add_test_result(self.last_test_result)
            self.state.save()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if not all(task.status == "completed" for task in tasks):
            return False
        if any(task.status == "failed" for task in tasks):
            return False
        if self.last_test_result is None:
            return False
        return self.last_test_result.passed

    def _get_tasks(self) -> list[Any]:
        return self.current_plan_tasks
