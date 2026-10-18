from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import (
    Plan,
    Settings,
    TaskModel,
    TestResult,
    settings as default_settings,
)
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger("furrow")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        send=None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.state_path = self.settings.workspace / ".furrow-state.json"
        self.send = send
        self.last_test_passed: bool | None = None

        logging.basicConfig(
            level=getattr(logging, self.settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    async def _emit(self, message: str) -> None:
        if self.send is None:
            return
        if asyncio.iscoroutinefunction(self.send):
            await self.send(message)
        else:
            self.send(message)

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        await self._emit(f"Started: {self.goal}")
        max_cycles = self.settings.max_cycles
        try:
            while max_cycles == 0 or self.cycles < max_cycles:
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._emit(f"Cycle {self.cycles}")
                logger.info("Starting cycle %d", self.cycles)
                await self._cycle()
                self._save_state()
                if self._is_done():
                    await self._emit("Goal complete. Halting.")
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
            else:
                await self._emit("Max cycles reached. Halting.")
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
        except Exception as e:  # pragma: no cover - surface errors to the caller
            await self._emit(f"Error: {e}")
            logger.exception("Orchestrator run failed")
            raise

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("No tasks planned.")
            return

        # Merge new tasks into the tracked task list (dedupe by id).
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)
                existing_ids.add(task.id)

        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            console.print("[yellow]All tasks already handled.[/yellow]")
            return

        max_parallel = self.settings.max_parallel_tasks or len(pending)
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_task(task: TaskModel) -> Any:
            async with semaphore:
                worker = WorkerAgent(
                    task=task, client=self.client, settings=self.settings
                )
                return await worker.run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[run_task(t) for t in pending], return_exceptions=True
            )

        for task, result in zip(pending, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("Task %s failed: %s", task.id, result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result.summary if hasattr(result, "summary") else str(result)
                logger.info("Task %s completed", task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"Task {task.id} completed: {task.result}")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed: %s", test_result.summary)
            self.last_test_passed = True
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            logger.warning("Tests failed: %s", test_result.summary)
            self.last_test_passed = False
            await self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                await self._emit(f"  • {failure}")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        pending = sum(1 for t in self.tasks if t.status == "pending")
        if pending > 0:
            return False
        # If the last test run failed, keep going so the new fix goal gets planned.
        if self.last_test_passed is False:
            return False
        return True

    def _save_state(self) -> None:
        try:
            payload = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
            }
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(payload, indent=2))
        except Exception as e:  # pragma: no cover - best effort persistence
            logger.warning("Failed to save state: %s", e)
