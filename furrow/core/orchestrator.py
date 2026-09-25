from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: Path | None = None,
        websocket=None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_file = state_file or (settings.workspace / ".furrow_state.json")
        self.websocket = websocket
        self._load_state()
        logger.info("orchestrator_initialized", goal=goal, state_file=str(self.state_file))

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                console.print(
                    f"[dim]Loaded state: {len(self.tasks)} tasks, cycle {self.cycles}[/dim]"
                )
                logger.info("state_loaded", tasks=len(self.tasks), cycles=self.cycles)
            except OSError as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("state_saved", tasks=len(self.tasks))
        except OSError as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")
            logger.error("state_save_failed", error=str(e))

    async def _send(self, message: str) -> None:
        if self.websocket:
            try:
                await self.websocket.send_text(message)
            except Exception as e:  # noqa: BLE001
                logger.debug("websocket_send_failed", error=str(e))
        console.print(message)

    async def run(self) -> None:
        await self._send(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                await self._send(
                    f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]"
                )
                logger.info("max_cycles_reached", max_cycles=settings.max_cycles)
                break

            self.cycles += 1
            await self._send(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            self._save_state()

            if self._is_done():
                await self._send("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        await self._send("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal, self.tasks)
        await self._send(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            await self._send("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing, preserving status of incomplete tasks
        task_map = {t.id: t for t in self.tasks}
        for new_task in plan.tasks:
            if new_task.id in task_map and task_map[new_task.id].status == "completed":
                # Preserve status if already completed
                new_task.status = "completed"
                new_task.result = task_map[new_task.id].result
            task_map[new_task.id] = new_task
        self.tasks = list(task_map.values())

        # Filter to only pending/in-progress tasks for execution
        pending_tasks = [t for t in self.tasks if t.status in ("pending", "failed")]
        if not pending_tasks:
            await self._send("[yellow]All tasks completed.[/yellow]")
            return

        await self._send("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        worker_tasks = [WorkerAgent(task=task, client=self.client).run() for task in pending_tasks]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._send(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                await self._send(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        await self._send("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            await self._send(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            await self._send(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._send(f"  • {failure}")
            await self._send("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures)
            # Update goal to focus on fixing failures
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)
