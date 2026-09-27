from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.text import Text

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        message_queue: asyncio.Queue[str] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.test_result: TestResult | None = None
        self._fix_context: str = ""
        self.message_queue = message_queue

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            )
        )
        max_cycles = self.client.settings.max_cycles
        while True:
            if max_cycles and self.cycles >= max_cycles:
                self._log(
                    f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]"
                )
                break
            self.cycles += 1
            self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._log("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self._planning_input())
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks
        if not plan.tasks:
            self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks...", console=console):
            semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

            async def _execute(task: TaskModel) -> str:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            results = await asyncio.gather(
                *(_execute(t) for t in plan.tasks), return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                self._log(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._log(f"[green]Task {task.id} completed[/green]")

        self._resolve_dependencies(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.original_goal, plan.tasks
            )
        self.test_result = test_result

        if test_result.passed:
            self._log(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._log(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._log(f"  • {failure}")
            self._log("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._fix_context = "\n".join(test_result.failures)

    def _resolve_dependencies(self, tasks: list[TaskModel]) -> None:
        task_map = {t.id: t for t in tasks}
        for task in tasks:
            if task.status == "failed":
                continue
            for dep_id in task.dependencies:
                dep = task_map.get(dep_id)
                if dep is not None and dep.status == "failed":
                    task.status = "failed"
                    task.result = "dependency failed"
                    self._log(
                        f"[red]Task {task.id} failed: dependency '{dep_id}' failed[/red]"
                    )
                    break

    def _planning_input(self) -> str:
        if self._fix_context:
            return f"{self.original_goal}\n\nPrevious failures to fix:\n{self._fix_context}"
        return self.original_goal

    def _is_done(self) -> bool:
        if self.test_result is not None:
            return self.test_result.passed
        tasks = self._get_tasks()
        if not tasks:
            return True  # No tasks planned = goal is complete
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _log(self, message: str, style: str | None = None) -> None:
        console.print(message, style=style)
        if self.message_queue is not None:
            try:
                plain = Text.from_markup(message).plain
            except Exception:
                plain = message
            try:
                self.message_queue.put_nowait(plain)
            except asyncio.QueueFull:
                pass
