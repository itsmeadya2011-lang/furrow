from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel
from furrow.llm import LLMClient
from furrow.logger import get_logger

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._all_tasks: list[TaskModel] = []
        self._completed_ids: set[str] = set()
        self._failed_ids: set[str] = set()
        self._stream_callback = None

    def set_stream_callback(self, callback) -> None:
        """Set an async callback for streaming output to websockets."""
        self._stream_callback = callback

    async def _stream(self, message: str) -> None:
        """Send a message via the stream callback if set."""
        if self._stream_callback:
            try:
                await self._stream_callback(message)
            except Exception:
                pass

    async def run(self) -> None:
        await self._stream(f"Furrow starting...\nGoal: {self.goal}")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            cycle_msg = f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
            console.print(cycle_msg)
            await self._stream(f"Cycle {self.cycles}")
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                await self._stream("Goal complete.")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles, max=max_cycles)
                await self._stream(f"Reached max cycles ({max_cycles}).")
                break

    async def run_with_stream(self, callback) -> None:
        """Run with a streaming callback for websocket output."""
        self.set_stream_callback(callback)
        await self.run()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._stream(f"Plan: {len(plan.tasks)} tasks")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._stream("No tasks planned.")
            return

        # Track all tasks across cycles
        for task in plan.tasks:
            if task.id not in {t.id for t in self._all_tasks}:
                self._all_tasks.append(task)

        # Order tasks by dependencies (topological sort)
        ordered = self._topological_sort(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in ordered
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(ordered, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._failed_ids.add(task.id)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._stream(f"Task {task.id} failed: {result}")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self._completed_ids.add(task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._stream(f"Task {task.id} completed")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._stream(f"Tests passed: {test_result.summary}")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._stream(f"Tests failed: {test_result.summary}")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            logger.warning("tests_failed", failures=test_result.failures)

    def _is_done(self) -> bool:
        completed = len(self._completed_ids)
        failed = len(self._failed_ids)
        if failed > 0:
            return False
        if not self._all_tasks:
            return True
        if completed >= len(self._all_tasks):
            return True
        return False

    def _topological_sort(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Sort tasks so dependencies run first."""
        task_map = {t.id: t for t in tasks}
        visited: set[str] = set()
        result: list[TaskModel] = []

        def visit(task: TaskModel) -> None:
            if task.id in visited:
                return
            visited.add(task.id)
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    visit(task_map[dep_id])
            result.append(task)

        for task in tasks:
            visit(task)

        return result

    def _get_tasks(self) -> list[TaskModel]:
        return self._all_tasks
