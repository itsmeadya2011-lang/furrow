from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger()
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = self.client.settings.max_cycles
        self.tasks: list[Any] = []
        self._last_test_result: TestResult | None = None
        self._on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None

    def on_event(
        self, callback: Callable[[str, dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._on_event = callback

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event is not None:
            await self._on_event(event_type, data)

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal)
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        await self._emit("start", {"goal": self.goal})

        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit("cycle_start", {"cycle": self.cycles})

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("complete", {"cycles": self.cycles})
                break

            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print(
                    "[yellow]Max cycles reached. Halting.[/yellow]"
                )
                await self._emit(
                    "max_cycles_reached",
                    {"cycles": self.cycles, "max_cycles": self.max_cycles},
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        logger.info("plan", plan=plan.model_dump())
        await self._emit("plan", {"plan": plan.model_dump()})

        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)
                existing_ids.add(task.id)
                console.print(f"[blue]Task {task.id} created[/blue]")
                await self._emit("task_created", {"task": task.model_dump()})

        if not plan.tasks:
            self._last_test_result = TestResult(
                passed=True,
                summary="No tasks planned - goal may be complete",
                failures=[],
            )
            console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            await self._emit("no_tasks", {"cycle": self.cycles})
            return

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=console
        ):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            existing = next(
                (t for t in self.tasks if t.id == task.id), task
            )
            if existing is task:
                self.tasks.append(task)
            if isinstance(result, Exception):
                existing.status = "failed"
                existing.result = str(result)
                console.print(
                    f"[red]Task {existing.id} failed: {result}[/red]"
                )
                logger.info(
                    "task_failed",
                    task_id=existing.id,
                    error=str(result),
                )
                await self._emit(
                    "task_failed",
                    {"task_id": existing.id, "error": str(result)},
                )
            else:
                existing.status = "completed"
                existing.result = result
                console.print(f"[green]Task {existing.id} completed[/green]")
                logger.info(
                    "task_completed",
                    task_id=existing.id,
                    result=result,
                )
                await self._emit(
                    "task_completed",
                    {"task_id": existing.id, "result": result},
                )

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(
                client=self.client
            ).run(self.goal, plan.tasks)
        self._last_test_result = test_result
        await self._emit("test", {"result": test_result.model_dump()})

        if test_result.passed:
            console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
            logger.info("test_passed", summary=test_result.summary)
        else:
            console.print(
                f"[red]Tests failed: {test_result.summary}[/red]"
            )
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print(
                "[yellow]Will attempt fix in next cycle.[/yellow]"
            )
            logger.info(
                "test_failed",
                summary=test_result.summary,
                failures=test_result.failures,
            )
            self.goal = f"Fix failing tests:\n" + "\n".join(
                test_result.failures
            )

    def _is_done(self) -> bool:
        if self._last_test_result is not None and self._last_test_result.passed:
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    def save_state(
        self, path: str | Path = ".furrow-state.json"
    ) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "max_cycles": self.max_cycles,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_state(
        self, path: str | Path = ".furrow-state.json"
    ) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text())
        self.goal = data.get("goal", self.goal)
        self.cycles = data.get("cycles", self.cycles)
        self.max_cycles = data.get("max_cycles", self.max_cycles)
        tasks_data = data.get("tasks", [])
        self.tasks = [TaskModel(**t) for t in tasks_data]
        return True
