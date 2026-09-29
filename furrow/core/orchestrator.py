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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

_default_console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = self.client.settings.max_cycles
        self.max_parallel = self.client.settings.max_parallel_tasks
        self._plan: Plan | None = None
        self.console = console or _default_console

    async def run(self) -> Plan | None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self.console.print(f"[yellow]Reached max_cycles ({self.max_cycles}). Halting.[/yellow]")
                break
        return self._plan

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._plan = plan
            return

        # Reset task statuses for a fresh cycle; preserve dependency info.
        for task in plan.tasks:
            task.status = "pending"

        self._plan = plan

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            results = await self._run_scheduled_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            elif isinstance(result, str):
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
            else:
                # Task was skipped because a dependency failed
                task.status = "skipped"
                task.result = str(result)
                self.console.print(f"[yellow]Task {task.id} skipped: {result}[/yellow]")

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  - {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_scheduled_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        """Execute tasks respecting dependencies, limited by max_parallel.

        Tasks whose dependencies have failed are skipped. Tasks are scheduled
        in waves: all tasks whose dependencies are completed run together.
        """
        sem = asyncio.Semaphore(self.max_parallel)
        results: dict[str, Any] = {}
        failed_ids: set[str] = set()

        async def _execute(task: TaskModel) -> str:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        # Topological scheduling by dependency waves.
        pending = list(tasks)
        max_rounds = len(tasks) + 1  # safety against deadlock cycles

        for _round in range(max_rounds):
            # Find tasks ready to run: all deps satisfied or skipped.
            ready = []
            still_pending = []
            for task in pending:
                if task.id in results:
                    continue
                deps = task.dependencies or []
                unresolved = [d for d in deps if d not in results]
                if not unresolved:
                    ready.append(task)
                else:
                    still_pending.append(task)

            if not ready:
                # Remaining tasks have unresolvable deps (cycle or skipped deps).
                for task in still_pending:
                    if task.id not in results:
                        results[task.id] = "Skipped due to unresolved dependencies"
                break

            # Check if any ready task has a failed dependency.
            runnable: list[TaskModel] = []
            for task in ready:
                deps = task.dependencies or []
                failed_deps = [d for d in deps if d in failed_ids]
                if failed_deps:
                    results[task.id] = f"Skipped: dependency failed ({', '.join(failed_deps)})"
                    failed_ids.add(task.id)
                else:
                    runnable.append(task)

            if runnable:
                coros = [_execute(t) for t in runnable]
                round_results = await asyncio.gather(*coros, return_exceptions=True)
                for task, result in zip(runnable, round_results):
                    results[task.id] = result
                    if isinstance(result, Exception):
                        failed_ids.add(task.id)

            pending = still_pending
            if not pending:
                break

        # Build results in original task order.
        return [results.get(t.id, "not executed") for t in tasks]

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        if self._plan is None:
            return []
        return self._plan.tasks
