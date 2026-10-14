from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.state import FurrowState, StateStore

logger = structlog.get_logger(__name__)
console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        message_callback: Callable[[dict], Awaitable[None]] | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self.message_callback = message_callback
        self.state_store = StateStore(path=state_path)
        self.state: FurrowState | None = None

    async def _emit(self, message: dict) -> None:
        if self.message_callback:
            await self.message_callback(message)

    async def run(self) -> None:
        logger.info("orchestrator.start", goal=self.goal)
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        await self._emit({"type": "cycle_start", "cycle": 0, "goal": self.goal})

        existing_state = self.state_store.load()
        if existing_state is not None:
            self.state = existing_state
            self.goal = existing_state.goal
            self.cycles = existing_state.cycles
            logger.info("state.resumed", goal=self.goal, cycles=self.cycles)
        else:
            self.state = FurrowState(goal=self.goal)
            logger.info("state.started", goal=self.goal)

        while True:
            self.cycles += 1
            logger.info("cycle.start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            self.state = FurrowState(
                goal=self.goal,
                cycles=self.cycles,
                tasks=self._tasks,
                last_test_result=self._last_test_result,
                status="running",
            )
            self.state_store.save(self.state)

            if self._is_done():
                logger.info("goal.complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit({"type": "goal_complete", "cycles": self.cycles})
                self.state.status = "complete"
                self.state_store.save(self.state)
                self.state_store.delete()
                break
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                logger.info("max_cycles.reached", cycles=self.cycles, max_cycles=max_cycles)
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                await self._emit(
                    {"type": "max_cycles_reached", "cycles": self.cycles, "max_cycles": max_cycles}
                )
                self.state.status = "stopped"
                self.state_store.save(self.state)
                break

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _plan_with_retry(self, goal: str) -> Plan:
        logger.info("planning.start", goal=goal)
        plan = await self.planner.plan(goal)
        logger.info("planning.complete", task_count=len(plan.tasks))
        return plan

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _test_with_retry(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing.start", goal=goal, task_count=len(tasks))
        test_result = await TesterAgent(client=self.client).run(goal, tasks)
        logger.info("testing.complete", passed=test_result.passed, summary=test_result.summary)
        return test_result

    async def _cycle(self) -> None:
        plan = await self._plan_with_retry(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(
            {"type": "plan", "cycle": self.cycles, "tasks": [t.model_dump() for t in plan.tasks]}
        )

        if not plan.tasks:
            logger.info("no_tasks_planned", goal=self.goal)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._tasks = plan.tasks
        logger.info("task_execution.start", task_count=len(self._tasks))

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run() for task in self._tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(self._tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task.failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(
                    {
                        "type": "task_complete",
                        "task_id": task.id,
                        "status": "failed",
                        "result": str(result),
                    }
                )
            else:
                task.status = "completed"
                task.result = result
                logger.info("task.completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(
                    {
                        "type": "task_complete",
                        "task_id": task.id,
                        "status": "completed",
                        "result": result,
                    }
                )

        self._last_test_result = await self._test_with_retry(self.goal, self._tasks)

        if self._last_test_result.passed:
            logger.info("tests.passed", summary=self._last_test_result.summary)
            console.print(f"[green]Tests passed: {self._last_test_result.summary}[/green]")
        else:
            logger.warning(
                "tests.failed",
                summary=self._last_test_result.summary,
                failures=self._last_test_result.failures,
            )
            console.print(f"[red]Tests failed: {self._last_test_result.summary}[/red]")
            for failure in self._last_test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(self._last_test_result.failures)

        await self._emit(
            {
                "type": "test_result",
                "cycle": self.cycles,
                "passed": self._last_test_result.passed,
                "summary": self._last_test_result.summary,
                "failures": self._last_test_result.failures,
            }
        )

    def _is_done(self) -> bool:
        if not self._tasks:
            return True

        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")

        if failed > 0:
            return False

        if completed >= len(self._tasks):
            if self._last_test_result is not None:
                return self._last_test_result.passed
            return True

        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
