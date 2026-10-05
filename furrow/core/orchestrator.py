from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.on_event = on_event
        self.plan: Plan | None = None
        self._failed_task_ids: set[str] = set()
        self._configure_logging()
        self._load_state()

    def _configure_logging(self) -> None:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(self.client.settings.log_level.upper())
            ),
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
        )
        self.logger = structlog.get_logger()

    def _load_state(self) -> None:
        state_file = self.client.settings.state_file
        if not state_file.exists():
            return
        try:
            with state_file.open("r") as f:
                data = json.load(f)
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            self._failed_task_ids = set(data.get("failed_task_ids", []))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("state_load_failed", error=str(exc))

    def _save_state(self) -> None:
        state_file = self.client.settings.state_file
        tasks = self._get_tasks()
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "plan": self.plan.model_dump() if self.plan else None,
            "tasks": [
                {
                    "id": t.id,
                    "status": t.status,
                    "result": t.result,
                }
                for t in tasks
            ],
            "failed_task_ids": list(self._failed_task_ids),
        }
        try:
            with state_file.open("w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            self.logger.warning("state_save_failed", error=str(exc))

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(event, data)
        self.logger.debug("event", event_name=event, **data)

    async def run(self, max_cycles: int | None = None) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        effective_max = max_cycles if max_cycles is not None else self.client.settings.max_cycles
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle_start", {"cycle": self.cycles, "goal": self.goal})
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("done", {"cycles": self.cycles, "reason": "complete"})
                self._save_state()
                break
            if effective_max > 0 and self.cycles >= effective_max:
                console.print(f"[yellow]Stopped after {self.cycles} cycles.[/yellow]")
                self._emit("stopped", {"cycles": self.cycles, "reason": "max_cycles"})
                self._save_state()
                break
            self._save_state()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit("plan_ready", {"plan": plan.model_dump(), "rationale": plan.rationale})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit(
                "cycle_end",
                {
                    "cycle": self.cycles,
                    "completed": 0,
                    "failed": 0,
                    "total": 0,
                    "passed_tests": False,
                },
            )
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_with_semaphore(task: Any) -> Any:
            async with semaphore:
                self._emit(
                    "task_start",
                    {"task_id": task.id, "description": task.description, "files": task.files},
                )
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit("task_failed", {"task_id": task.id, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit("task_completed", {"task_id": task.id, "result": result})

        completed_count = sum(1 for t in plan.tasks if t.status == "completed")
        failed_count = sum(1 for t in plan.tasks if t.status == "failed")

        self._emit("testing", {"goal": self.goal})
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit("tests_passed", {"summary": test_result.summary})
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            self._emit(
                "tests_failed", {"summary": test_result.summary, "failures": test_result.failures}
            )

        self._emit(
            "cycle_end",
            {
                "cycle": self.cycles,
                "completed": completed_count,
                "failed": failed_count,
                "total": len(plan.tasks),
                "passed_tests": test_result.passed,
            },
        )

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed == 0 and completed == len(tasks):
            return True
        if failed > 0:
            current_failed_ids = {t.id for t in tasks if t.status == "failed"}
            if current_failed_ids.issubset(self._failed_task_ids):
                return True
            self._failed_task_ids.update(current_failed_ids)
        return False

    def _get_tasks(self) -> list[Any]:
        return self.plan.tasks if self.plan else []
