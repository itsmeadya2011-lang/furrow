from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        websocket: Any | None = None,
    ) -> None:
        self.goal = goal
        self.config = config or settings
        self.client = client or LLMClient(settings=self.config)
        self.planner = PlannerAgent(client=self.client, config=self.config)
        self.cycles = 0
        self.task_history: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.websocket = websocket

    async def run(self) -> None:
        await self._send(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", "info")
        logger.info("orchestrator_started", goal=self.goal, max_cycles=self.config.max_cycles)

        while True:
            if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                await self._send(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Halting.[/yellow]", "warning")
                logger.info("max_cycles_reached", cycles=self.cycles)
                break

            self.cycles += 1
            await self._send(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]", "info")
            await self._cycle()

            if self._is_done():
                await self._send("[bold green]Goal complete. Halting.[/bold green]", "success")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        cycle_start = time.time()
        await self._send("[bold yellow]Planning...[/bold yellow]", "info")

        try:
            plan = await self.planner.plan(self.goal, self.task_history)
            await self._send(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"), "info")
            logger.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks))

            if not plan.tasks:
                await self._send("[yellow]No tasks planned. Goal may be complete.[/yellow]", "warning")
                return

            # Enforce max_parallel_tasks
            tasks_to_run = plan.tasks[: self.config.max_parallel_tasks]
            if len(plan.tasks) > self.config.max_parallel_tasks:
                await self._send(
                    f"[yellow]Limiting to {self.config.max_parallel_tasks} parallel tasks (planned: {len(plan.tasks)})[/yellow]",
                    "warning",
                )

            await self._send("[bold yellow]Executing tasks in parallel...[/bold yellow]", "info")

            worker_tasks = [
                WorkerAgent(task=task, client=self.client, config=self.config).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            cycle_tasks: list[TaskModel] = []
            for task, result in zip(tasks_to_run, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    await self._send(f"[red]Task {task.id} failed: {result}[/red]", "error")
                    logger.error("task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    await self._send(f"[green]Task {task.id} completed[/green]", "success")
                    logger.info("task_completed", task_id=task.id)
                cycle_tasks.append(task)

            self.task_history.extend(cycle_tasks)

            await self._send("[bold yellow]Testing...[/bold yellow]", "info")
            test_result = await TesterAgent(client=self.client, config=self.config).run(
                self.goal, cycle_tasks
            )

            cycle_duration = time.time() - cycle_start
            self.cycle_history.append(
                {
                    "cycle": self.cycles,
                    "tasks": [t.model_dump() for t in cycle_tasks],
                    "test_result": test_result.model_dump(),
                    "duration_seconds": cycle_duration,
                }
            )

            if test_result.passed:
                await self._send(f"[green]Tests passed: {test_result.summary}[/green]", "success")
                logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            else:
                await self._send(f"[red]Tests failed: {test_result.summary}[/red]", "error")
                for failure in test_result.failures:
                    await self._send(f"  • {failure}", "error")
                await self._send("[yellow]Will attempt fix in next cycle.[/yellow]", "warning")
                logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
                self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        except BaseException as e:  # noqa: BLE001
            logger.error("cycle_failed", cycle=self.cycles, error=str(e))
            await self._send(f"[red]Cycle {self.cycles} failed: {e}[/red]", "error")
            # Don't re-raise, continue to next cycle

    def _is_done(self) -> bool:
        if not self.task_history:
            return False

        completed = sum(1 for t in self.task_history if t.status == "completed")
        failed = sum(1 for t in self.task_history if t.status == "failed")
        pending = sum(1 for t in self.task_history if t.status == "pending")

        if failed > 0 and pending == 0:
            return False
        return bool(completed >= len(self.task_history) and pending == 0)

    async def _send(self, content: Any, level: str = "info") -> None:
        if self.websocket:
            try:
                await self.websocket.send_json({"level": level, "content": str(content)})
            except OSError:
                # WebSocket disconnected, ignore
                pass
        else:
            console.print(content)
