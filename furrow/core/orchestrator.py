from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        if settings is not None:
            self.settings = settings
        elif client is not None and hasattr(client, "settings"):
            self.settings = client.settings
        else:
            self.settings = Settings()
        self.max_cycles = max(0, self.settings.max_cycles)
        self.max_parallel_tasks = max(1, self.settings.max_parallel_tasks)
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self._has_work_started = False
        self._latest_test_passed: bool | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while self.max_cycles == 0 or self.cycles < self.max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        if self.max_cycles > 0 and self.cycles >= self.max_cycles and not self._is_done():
            console.print(
                f"[yellow]Maximum cycle limit ({self.max_cycles}) reached. Halting.[/yellow]"
            )

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.tasks = plan.tasks
        self._has_work_started = True
        self._latest_test_passed = False
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if self.tasks:
            semaphore = asyncio.Semaphore(self.max_parallel_tasks)

            async def run_worker(task: TaskModel) -> Any:
                async with semaphore:
                    try:
                        return await WorkerAgent(
                            task=task,
                            client=self.client,
                            settings=self.settings,
                        ).run()
                    except Exception as exc:
                        return exc

            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                workers = [asyncio.create_task(run_worker(task)) for task in self.tasks]
                results = await asyncio.gather(*workers)

            for task, result in zip(self.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
        else:
            console.print("[yellow]No tasks planned. Running validation only.[/yellow]")

        with Status("[bold yellow]Testing...", console=console):
            try:
                test_result = await TesterAgent(
                    client=self.client,
                    settings=self.settings,
                ).run(self.goal, self.tasks)
            except Exception as exc:
                test_result = TestResult(
                    passed=False,
                    summary=str(exc),
                    failures=[str(exc)],
                )

        if test_result.passed:
            self._latest_test_passed = True
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._latest_test_passed = False
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Original goal: {self.original_goal}\n\nFix failing tests:\n" + "\n".join(
                test_result.failures
            )

    def _is_done(self) -> bool:
        if not self._has_work_started:
            return False
        if any(task.status != "completed" for task in self.tasks):
            return False
        return self._latest_test_passed is True

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
