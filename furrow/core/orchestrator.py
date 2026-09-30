from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import aiofiles
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, WorkerResult
from furrow.llm import LLMClient

logger = structlog.get_logger("furrow.orchestrator")

console = Console()

Reporter = Callable[[str, dict[str, Any]], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.cycle_history: list[dict[str, Any]] = []
        self.current_plan: Plan | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._test_result: TestResult | None = None
        self.reporter: Reporter | None = None

    async def run(self, reporter: Reporter | None = None) -> None:
        self.reporter = reporter
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        logger.info("orchestrator_start", goal=self.goal)
        if self.reporter:
            await self.reporter("start", {"goal": self.goal})

        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                logger.info("max_cycles_reached", cycles=self.cycles)
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

            await self._save_state()

    async def _cycle(self) -> dict[str, Any]:
        plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale[:100])
        if self.reporter:
            await self.reporter("plan", {"plan": plan.model_dump()})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return {"plan": plan.model_dump(), "tasks": [], "test": None}

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [self._run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            elif isinstance(result, WorkerResult):
                task.status = "completed"
                task.result = result.summary
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id, summary=result.summary[:100])
            else:
                task.status = "completed"
                task.result = str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary[:100])
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        if self.reporter:
            await self.reporter(
                "cycle",
                {
                    "cycle": self.cycles,
                    "plan": plan.model_dump(),
                    "tasks": [t.model_dump() for t in plan.tasks],
                    "test": test_result.model_dump(),
                },
            )

        cycle_data = {
            "cycle": self.cycles,
            "plan": plan.model_dump(),
            "tasks": [t.model_dump() for t in plan.tasks],
            "test": test_result.model_dump(),
        }
        self.cycle_history.append(cycle_data)
        return cycle_data

    async def _run_task(self, task: TaskModel) -> WorkerResult:
        assert self._semaphore is not None
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        # Enforce max_cycles
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            logger.info("max_cycles_reached_is_done")
            return True

        # If tests failed in the last cycle, keep working to fix them
        if self._test_result is not None and not self._test_result.passed:
            return False

        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            return False

        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        if self.current_plan is None:
            return []
        return self.current_plan.tasks

    async def _save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "cycle_history": self.cycle_history,
            "current_plan": self.current_plan.model_dump() if self.current_plan else None,
        }
        state_path = self.settings.workspace / ".furrow_state.json"
        try:
            async with aiofiles.open(state_path, "w") as f:
                await f.write(json.dumps(state, indent=2, default=str))
            logger.info("state_saved", path=str(state_path))
        except OSError as e:
            logger.warning("state_save_failed", error=str(e))
