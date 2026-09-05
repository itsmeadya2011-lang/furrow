from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import aiofiles
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)

STATE_FILE = ".furrow/state.json"


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.last_plan: Plan | None = None
        self.settings = Settings()
        self.status_callback = status_callback
        self._state_path = Path(self.settings.workspace) / STATE_FILE

    async def run(self) -> None:
        await self._load_state()
        if self.status_callback:
            self.status_callback("Starting orchestrator")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self.status_callback:
                self.status_callback(f"Cycle {self.cycles}: planning")
            await self._cycle()
            await self._save_state()
            if self.status_callback:
                self.status_callback(f"Cycle {self.cycles}: complete")
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                if self.status_callback:
                    self.status_callback("Goal complete")
                break

    async def _cycle(self) -> None:
        if self.status_callback:
            self.status_callback("Planning...")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.last_plan = plan
        logger.info("Plan generated with %d tasks", len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("No tasks planned. Goal may be complete.")
            if self.status_callback:
                self.status_callback("No tasks planned")
            return

        if self.status_callback:
            self.status_callback(f"Executing {len(plan.tasks)} tasks")
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("Task %s failed: %s", task.id, result)
            else:
                task.status = "completed"
                task.result = result
                logger.info("Task %s completed", task.id)
            if self.status_callback:
                self.status_callback(f"Task {task.id} {task.status}")

        completed = sum(1 for t in plan.tasks if t.status == "completed")
        failed = sum(1 for t in plan.tasks if t.status == "failed")
        pending = sum(1 for t in plan.tasks if t.status == "pending")
        logger.info("Cycle %d complete: %d completed, %d failed, %d pending", self.cycles, completed, failed, pending)

        if self.status_callback:
            self.status_callback("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("Tests passed: %s", test_result.summary)
            if self.status_callback:
                self.status_callback(f"Tests passed: {test_result.summary}")
        else:
            logger.error("Tests failed: %s", test_result.summary)
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            logger.info("Will attempt fix in next cycle.")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            if self.status_callback:
                self.status_callback(f"Tests failed: {test_result.summary}")

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.last_plan.tasks if self.last_plan else []

    async def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            async with aiofiles.open(self._state_path, "r") as f:
                data = json.loads(await f.read())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            logger.info("Loaded state from %s (cycle %d)", self._state_path, self.cycles)
        except Exception as e:
            logger.warning("Failed to load state: %s", e)

    async def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tasks = []
            if self.last_plan:
                tasks = [
                    {
                        "id": t.id,
                        "description": t.description,
                        "files": t.files,
                        "dependencies": t.dependencies,
                        "status": t.status,
                        "result": t.result,
                    }
                    for t in self.last_plan.tasks
                ]
            payload = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": tasks,
            }
            async with aiofiles.open(self._state_path, "w") as f:
                await f.write(json.dumps(payload, indent=2))
        except Exception as e:
            logger.warning("Failed to save state: %s", e)
