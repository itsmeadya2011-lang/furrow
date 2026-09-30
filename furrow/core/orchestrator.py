from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self, goal: str, client: LLMClient | None = None, max_cycles: int = 0
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.max_cycles = max_cycles

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._incorporate_tasks(plan.tasks)

        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks...", console=console):
            await self._execute_tasks()

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _incorporate_tasks(self, plan_tasks: list[TaskModel]) -> None:
        """Merge planned tasks into self.tasks for tracking.

        Completed tasks from prior cycles are kept (never re-run) and are not
        overwritten by a fresh plan. Incomplete tasks from a previous cycle
        that are restated by the new plan are replaced; incomplete leftovers not
        restated by the plan are preserved.
        """
        merged_by_id: dict[str, TaskModel] = {}
        plan_ids = {t.id for t in plan_tasks}
        # Completed tasks from prior cycles are authoritative; keep them.
        for t in self.tasks:
            if t.status == "completed":
                merged_by_id[t.id] = t
        # Incorporate plan tasks; do not overwrite kept completed tasks.
        for t in plan_tasks:
            if t.id not in merged_by_id:
                merged_by_id[t.id] = t
        # Preserve incomplete tasks from a previous cycle not restated in the plan.
        for t in self.tasks:
            if t.id not in merged_by_id and t.id not in plan_ids:
                merged_by_id[t.id] = t
        self.tasks = list(merged_by_id.values())

    async def _execute_tasks(self) -> None:
        max_parallel = max(1, self.client.settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        while True:
            remaining = [t for t in self.tasks if t.status == "pending"]
            if not remaining:
                break
            ready = self._get_ready_tasks()
            if not ready:
                blocked = [t for t in remaining if self._has_failed_dependency(t)]
                if blocked:
                    console.print("[red]Tasks blocked by failed dependencies. Will attempt fix in next cycle.[/red]")
                else:
                    console.print("[yellow]No ready tasks. Halting task execution.[/yellow]")
                break
            results = await asyncio.gather(
                *[_run(t) for t in ready], return_exceptions=True
            )
            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")

    def _get_ready_tasks(self) -> list[TaskModel]:
        """Return pending tasks whose dependencies are all completed."""
        completed_ids = {t.id for t in self.tasks if t.status == "completed"}
        ready: list[TaskModel] = []
        for task in self.tasks:
            if task.status != "pending":
                continue
            deps = task.dependencies or []
            if all(dep in completed_ids for dep in deps):
                ready.append(task)
        return ready

    def _has_failed_dependency(self, task: TaskModel) -> bool:
        failed_ids = {t.id for t in self.tasks if t.status == "failed"}
        return any(dep in failed_ids for dep in (task.dependencies or []))

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        all_completed = all(t.status == "completed" for t in tasks)
        none_failed = not any(t.status == "failed" for t in tasks)
        return all_completed and none_failed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
