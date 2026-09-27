from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

_console = Console()
console = _console


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.console = console if console is not None else _console
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                self.console.print(
                    f"[yellow]Reached max_cycles limit ({max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute_plan(plan)

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_plan(self, plan: Plan) -> None:
        max_parallel = max(1, self.client.settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(max_parallel)
        task_map: dict[str, TaskModel] = {t.id: t for t in plan.tasks}
        all_ids: set[str] = set(task_map)
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()
        terminal: set[str] = set()

        for task in plan.tasks:
            if task.status == "completed":
                completed_ids.add(task.id)
                terminal.add(task.id)
            elif task.status == "failed":
                failed_ids.add(task.id)
                terminal.add(task.id)

        async def _run(task: TaskModel) -> str:
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client)
                return await worker.run()

        while len(terminal) < len(plan.tasks):
            ready: list[TaskModel] = []
            failed_now: list[TaskModel] = []
            for task in plan.tasks:
                if task.id in terminal:
                    continue
                in_plan_deps = [d for d in task.dependencies if d in all_ids]
                failed_deps = [d for d in in_plan_deps if d in failed_ids]
                if failed_deps:
                    task.status = "failed"
                    task.result = f"Dependency failed: {', '.join(failed_deps)}"
                    failed_now.append(task)
                    continue
                unmet = [d for d in in_plan_deps if d not in completed_ids]
                if not unmet:
                    ready.append(task)

            for task in failed_now:
                failed_ids.add(task.id)
                terminal.add(task.id)
                self.console.print(f"[red]Task {task.id} failed: dependency failed[/red]")

            if not ready:
                for task in plan.tasks:
                    if task.id not in terminal:
                        task.status = "failed"
                        task.result = "Blocked: unmet or cyclic dependencies"
                        failed_ids.add(task.id)
                        terminal.add(task.id)
                        self.console.print(f"[red]Task {task.id} failed: blocked[/red]")
                break

            with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
                outcomes = await asyncio.gather(*[_run(t) for t in ready], return_exceptions=True)

            for task, outcome in zip(ready, outcomes):
                if isinstance(outcome, Exception):
                    task.status = "failed"
                    task.result = str(outcome)
                    failed_ids.add(task.id)
                    terminal.add(task.id)
                    self.console.print(f"[red]Task {task.id} failed: {outcome}[/red]")
                else:
                    task.status = "completed"
                    task.result = outcome
                    completed_ids.add(task.id)
                    terminal.add(task.id)
                    self.console.print(f"[green]Task {task.id} completed[/green]")

    def _get_tasks(self) -> list[TaskModel]:
        if self.current_plan is None:
            return []
        return self.current_plan.tasks

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        all_completed = all(t.status == "completed" for t in tasks)
        if any(t.status == "failed" for t in tasks):
            return False
        if not all_completed:
            return False
        return self.last_test_result is not None and self.last_test_result.passed
