from __future__ import annotations

import asyncio
import json
import os
import re


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
        output_queue: asyncio.Queue[str] | None = None,
    ) -> None:
        self.original_goal = goal
        self.current_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.last_plan: Plan | None = None
        self.output_queue = output_queue

    async def run(self) -> None:
        self._emit("Furrow\nGoal: {self.original_goal}")
        while True:
            self.cycles += 1
            self._emit(f"\n═══ Cycle {self.cycles} ═══")
            await self._cycle()
            if self._is_done():
                self._emit("Goal complete. Halting.")
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self._emit(f"Reached max_cycles={self.client.settings.max_cycles}. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.current_goal)
        self.last_plan = plan
        self._emit_panel(plan.model_dump(), "Plan", "blue")

        if not plan.tasks:
            self._emit("No tasks planned. Goal may be complete.")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            parallel_tasks = plan.tasks[: self.client.settings.max_parallel_tasks]
            if len(parallel_tasks) < len(plan.tasks):
                self._emit(
                    f"Limiting to {len(parallel_tasks)} of {len(plan.tasks)} tasks (max_parallel_tasks)."
                )
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in parallel_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.current_goal, plan.tasks)

        if test_result.passed:
            self._emit(f"Tests passed: {test_result.summary}")
        else:
            self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("Will attempt fix in next cycle.")
            self.current_goal = (
                f"Original goal: {self.original_goal}\n\n"
                f"Fix failing tests from cycle {self.cycles}:\n" + "\n".join(test_result.failures)
            )

    def _is_done(self) -> bool:
        if self.last_plan is None or not self.last_plan.tasks:
            return True
        completed = sum(1 for t in self.last_plan.tasks if t.status == "completed")
        failed = sum(1 for t in self.last_plan.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.last_plan.tasks):
            return True
        return False

    def _get_tasks(self) -> list:
        return self.last_plan.tasks if self.last_plan else []

    def _emit(self, message: str) -> None:
        console.print(message)
        if self.output_queue is not None:
            clean = re.sub(r"\[/?[a-zA-Z0-9 ]+\]", "", str(message))
            self.output_queue.put_nowait(clean)

    def _emit_panel(self, content: str, title: str, border_style: str) -> None:
        panel = Panel(Pretty(content), title=title, border_style=border_style)
        console.print(panel)
        if self.output_queue is not None:
            clean = re.sub(r"\[/?[a-zA-Z0-9 ]+\]", "", str(content))
            self.output_queue.put_nowait(f"[{title}] {clean}")
