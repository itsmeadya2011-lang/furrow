from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], None] | None = None,
        max_cycles: int = 0,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self._output_callback = output_callback
        self._all_tasks: dict[str, TaskModel] = {}
        self._completed_ids: set[str] = set()
        self._failed_ids: set[str] = set()

    def _emit(self, text: str) -> None:
        if self._output_callback:
            result = self._output_callback(text)
            if asyncio.iscoroutine(result):
                asyncio.get_running_loop().create_task(result)

    async def run(self) -> None:
        msg = f"Furrow\nGoal: {self.goal}"
        console.print(Panel.fit(msg, title="Furrow"))
        self._emit(msg)
        while True:
            self.cycles += 1
            msg = f"Cycle {self.cycles}"
            console.print(f"\n{'═' * 40}\n{msg}\n{'═' * 40}")
            self._emit(msg)

            if self.max_cycles > 0 and self.cycles > self.max_cycles:
                msg = f"Reached max cycles ({self.max_cycles}). Stopping."
                console.print(f"[bold yellow]{msg}[/bold yellow]")
                self._emit(msg)
                break

            await self._cycle()

            if self._is_done():
                msg = "Goal complete. Halting."
                console.print(f"[bold green]{msg}[/bold green]")
                self._emit(msg)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)

        for task in plan.tasks:
            self._all_tasks[task.id] = task

        msg = f"Plan:\n{plan.model_dump_json(indent=2)}"
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(msg)

        if not plan.tasks:
            pending = [t for t in self._all_tasks.values() if t.status == "pending"]
            if not pending:
                msg = "No new tasks planned and all existing tasks complete."
                console.print(f"[green]{msg}[/green]")
                self._emit(msg)
            else:
                msg = "No new tasks planned. Goal may be complete."
                console.print(f"[yellow]{msg}[/yellow]")
                self._emit(msg)
            return

        ordered_tasks = self._order_tasks(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in ordered_tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(ordered_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._failed_ids.add(task.id)
                msg = f"Task {task.id} failed: {result}"
                console.print(f"[red]{msg}[/red]")
                self._emit(msg)
            else:
                task.status = "completed"
                task.result = result
                self._completed_ids.add(task.id)
                msg = f"Task {task.id} completed"
                console.print(f"[green]{msg}[/green]")
                self._emit(msg)

        failed = [t for t in ordered_tasks if isinstance(
            results[ordered_tasks.index(t)], Exception
        )]
        if failed:
            msg = f"{len(failed)} task(s) failed. Will retry in next cycle."
            console.print(f"[yellow]{msg}[/yellow]")
            self._emit(msg)
            for t in failed:
                t.status = "pending"
                self._failed_ids.discard(t.id)
            self.goal = "Retry failed tasks:\n" + "\n".join(
                f"Task {t.id}: {t.description}" for t in failed
            )
            return

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, list(self._all_tasks.values())
            )

        if test_result.passed:
            msg = f"Tests passed: {test_result.summary}"
            console.print(f"[green]{msg}[/green]")
            self._emit(msg)
        else:
            msg = f"Tests failed: {test_result.summary}"
            console.print(f"[red]{msg}[/red]")
            self._emit(msg)
            for failure in test_result.failures:
                msg = f"  • {failure}"
                console.print(msg)
                self._emit(msg)
            msg = "Will attempt fix in next cycle."
            console.print(f"[yellow]{msg}[/yellow]")
            self._emit(msg)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _order_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        ordered: list[TaskModel] = []
        remaining = list(tasks)
        while remaining:
            ready = [
                t for t in remaining
                if not t.dependencies or all(
                    dep in self._completed_ids for dep in t.dependencies
                )
            ]
            if not ready:
                ready = remaining
            for t in ready:
                ordered.append(t)
                remaining.remove(t)
        return ordered

    def _is_done(self) -> bool:
        pending = [t for t in self._all_tasks.values() if t.status == "pending"]
        failed = [t for t in self._all_tasks.values() if t.status == "failed"]
        if pending or failed:
            return False
        if not self._all_tasks:
            return False
        return all(t.status == "completed" for t in self._all_tasks.values())