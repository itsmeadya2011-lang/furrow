from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
        websocket: Any | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings_obj)
        self.settings = settings_obj or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.websocket = websocket
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        state_path = self.settings.workspace / STATE_FILE
        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                # Restore tasks from state
                for task_data in data.get("tasks", []):
                    from furrow.config import TaskModel
                    task = TaskModel(**task_data)
                    self.tasks.append(task)
            except (OSError, json.JSONDecodeError) as e:
                logger.debug("load_state_error", error=str(e))
                # Ignore corrupted state

    def _save_state(self) -> None:
        """Persist state to disk."""
        state_path = self.settings.workspace / STATE_FILE
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "tasks": [task.model_dump() for task in self.tasks],
        }
        try:
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)
        except (OSError, TypeError) as e:
            logger.debug("save_state_error", error=str(e))

    async def _send(self, message: str) -> None:
        """Send message to console and/or websocket."""
        console.print(message)
        logger.info("orchestrator_output", message=message)
        if self.websocket:
            try:
                await self.websocket.send_text(message)
            except (OSError, RuntimeError) as e:
                logger.debug("websocket_send_error", error=str(e))

    async def run(self) -> None:
        await self._send(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        logger.info("orchestrator_start", goal=self.goal)
        max_cycles = self.settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._send(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                logger.warning("max_cycles_reached", max_cycles=max_cycles)
                break
            self.cycles += 1
            await self._send(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                await self._send("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_complete", cycles=self.cycles)
                break
        self._save_state()

    async def _cycle(self) -> None:
        await self._send("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal)
        await self._send(f"[blue]Plan:[/blue] {plan.model_dump()}")
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            await self._send("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing, avoiding duplicates by id
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)

        await self._send("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        worker_tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
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

        self._save_state()

        await self._send("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._send(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            await self._send(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._send(f"  • {failure}")
            await self._send("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures)
            # Append failures to goal for next cycle without losing original goal
            self.goal = f"{self.goal}\n\nPrevious failures to fix:\n" + "\n".join(test_result.failures)

        self._save_state()

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)
