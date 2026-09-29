from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()

CycleEndCallback = Callable[[int, Plan], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_tasks: list[TaskModel] = []
        self._tests_passed = False

    async def run(self, on_cycle_end: CycleEndCallback | None = None) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            plan = await self._cycle()

            if on_cycle_end is not None:
                await on_cycle_end(self.cycles, plan)

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> Plan:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._current_tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._tests_passed = True
            return plan

        await self._execute_tasks(plan)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._tests_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return plan

    async def _execute_tasks(self, plan: Plan) -> None:
        tasks = plan.tasks
        id_to_task = {t.id: t for t in tasks}

        # Sanitize dependencies: drop self-references and references to unknown tasks.
        for t in tasks:
            t.dependencies = [d for d in t.dependencies if d in id_to_task and d != t.id]

        # Break cycles so the scheduler cannot deadlock. Tasks that participate in a
        # dependency cycle lose their in-cycle dependencies (treated as no-dependency).
        cycle_ids = self._find_cyclic_task_ids(tasks)
        for tid in cycle_ids:
            task = id_to_task[tid]
            task.dependencies = [d for d in task.dependencies if d not in cycle_ids]

        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()
        skipped_ids: set[str] = set()
        pending: set[str] = {t.id for t in tasks}

        async def run_one(task: TaskModel) -> None:
            blocked_by = [d for d in task.dependencies if d in failed_ids or d in skipped_ids]
            if blocked_by:
                task.status = "skipped"
                task.result = f"Skipped: dependency {', '.join(blocked_by)} did not complete"
                console.print(
                    f"[yellow]Task {task.id} skipped: blocked by failed dependency[/yellow]"
                )
                return

            async with semaphore:
                outcome = await WorkerAgent(task=task, client=self.client).run()

            task.status = "completed"
            task.result = outcome
            console.print(f"[green]Task {task.id} completed[/green]")

        while pending:
            processable = [
                t
                for t in tasks
                if t.id in pending
                and all(
                    d in completed_ids or d in failed_ids or d in skipped_ids
                    for d in t.dependencies
                )
            ]

            if not processable:
                # No runnable tasks remain despite pending work; break potential
                # deadlocks by skipping the rest rather than stalling forever.
                for t in tasks:
                    if t.id in pending:
                        t.status = "skipped"
                        t.result = "Skipped: unresolvable dependency state"
                        skipped_ids.add(t.id)
                        pending.discard(t.id)
                        console.print(f"[yellow]Task {t.id} skipped[/yellow]")
                break

            results = await asyncio.gather(
                *(run_one(t) for t in processable), return_exceptions=True
            )

            for task, result in zip(processable, results):
                pending.discard(task.id)
                if task.status == "completed":
                    completed_ids.add(task.id)
                elif task.status == "skipped":
                    skipped_ids.add(task.id)
                elif task.status == "failed":
                    failed_ids.add(task.id)
                elif isinstance(result, Exception):
                    # run_one normally catches worker errors internally, but guard
                    # against any unexpected exception surfacing through gather.
                    task.status = "failed"
                    task.result = str(result)
                    failed_ids.add(task.id)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")

    def _find_cyclic_task_ids(self, tasks: list[TaskModel]) -> set[str]:
        """Return the IDs of all tasks that participate in a dependency cycle.

        Uses Tarjan's strongly-connected-components algorithm. Self-loops are
        impossible here because dependencies referencing the task itself are
        dropped during sanitization.
        """
        id_to_task = {t.id: t for t in tasks}
        adj: dict[str, list[str]] = {t.id: [] for t in tasks}
        for t in tasks:
            for d in t.dependencies:
                if d in id_to_task:
                    adj[d].append(t.id)

        index_counter = [0]
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        sccs: list[set[str]] = []

        def strongconnect(v: str) -> None:
            indices[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)
            for w in adj[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                scc: set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)

        for tid in id_to_task:
            if tid not in indices:
                strongconnect(tid)

        cycle_ids: set[str] = set()
        for scc in sccs:
            if len(scc) > 1:
                cycle_ids |= scc
            else:
                only = next(iter(scc))
                if only in adj.get(only, []):
                    cycle_ids.add(only)
        return cycle_ids

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return bool(self._tests_passed)

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        skipped = sum(1 for t in tasks if t.status == "skipped")

        if failed > 0 or skipped > 0:
            return False

        if completed >= len(tasks):
            return bool(self._tests_passed)

        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._current_tasks
