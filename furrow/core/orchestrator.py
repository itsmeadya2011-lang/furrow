from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from inspect import isawaitable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()

OutputCallback = Callable[[str], Any]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.output_callback = output_callback
        self.cycles = 0
        self.tasks: list[Any] = []

    async def _emit(self, message: Any, output: str | None = None) -> None:
        console.print(message)
        if self.output_callback is None:
            return

        result = self.output_callback(output if output is not None else str(message))
        if isawaitable(result):
            await result

    async def run(self) -> None:
        header = f"Furrow\nGoal: {self.goal}"
        await self._emit(Panel.fit(header, title="Furrow"), header)
        while True:
            self.cycles += 1
            await self._emit(
                f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]",
                f"Cycle {self.cycles}",
            )
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]", "Goal complete. Halting.")
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                await self._emit(
                    f"[yellow]Reached max cycles ({self.client.settings.max_cycles}). Halting.[/yellow]",
                    f"Reached max cycles ({self.client.settings.max_cycles}). Halting.",
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        plan_data = plan.model_dump()
        await self._emit(
            Panel(Pretty(plan_data), title="Plan", border_style="blue"),
            f"Plan:\n{json.dumps(plan_data, indent=2)}",
        )

        self.tasks = list(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

            async def run_task(task: Any) -> Any:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            tasks = [run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                output = f"Task {task.id} failed: {result}"
                await self._emit(f"[red]{output}[/red]", output)
            else:
                task.status = "completed"
                task.result = result
                output = f"Task {task.id} completed"
                await self._emit(f"[green]{output}[/green]", output)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            output = f"Tests passed: {test_result.summary}"
            await self._emit(f"[green]{output}[/green]", output)
        else:
            output = f"Tests failed: {test_result.summary}"
            await self._emit(f"[red]{output}[/red]", output)
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit(
                "[yellow]Will attempt fix in next cycle.[/yellow]",
                "Will attempt fix in next cycle.",
            )
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False
