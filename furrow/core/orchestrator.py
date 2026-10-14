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
from furrow.config import Plan, Settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings_obj: Settings | None = None) -> None:
        self.goal = goal
        self.settings = settings_obj or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.plan: Plan | None = None
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Build context from previous cycles
        context = self._build_context()

        with Status("[bold yellow]Planning...", console=console) as _status:
            plan = await self.planner.plan(self.goal, context=context)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks respecting dependencies and max_parallel_tasks
        results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "plan": plan.model_dump(),
            "test_result": test_result.model_dump(),
        })

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[Any]) -> list[Any]:
        """Execute tasks with dependency resolution and parallelism limit."""
        # Build dependency graph
        deps = {task.id: set(task.dependencies) for task in tasks}
        completed: set[str] = set()
        results: list[Any] = [None] * len(tasks)
        running: dict[str, asyncio.Task] = {}
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task(task: Any) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

        while len(completed) < len(tasks):
            # Find tasks ready to run (dependencies met, not running, not done)
            ready = [
                task for task in tasks
                if task.id not in completed
                and task.id not in running
                and deps[task.id].issubset(completed)
            ]

            # Start ready tasks
            for task in ready:
                running[task.id] = asyncio.create_task(run_task(task))

            if not running:
                # No tasks running and not all done - circular dependency or all blocked
                remaining = [t for t in tasks if t.id not in completed]
                console.print(f"[red]Stalled: {len(remaining)} tasks blocked by unmet dependencies[/red]")
                for task in remaining:
                    task.status = "failed"
                    task.result = "Blocked by unmet dependencies"
                    results[tasks.index(task)] = Exception("Blocked by unmet dependencies")
                    completed.add(task.id)
                break

            # Wait for at least one to complete
            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                for task_id, task_fut in running.items():
                    if task_fut is fut:
                        idx = next(i for i, t in enumerate(tasks) if t.id == task_id)
                        results[idx] = fut.result() if not fut.cancelled() else Exception("Cancelled")
                        completed.add(task_id)
                        del running[task_id]
                        break

        return results

    def _build_context(self) -> str:
        """Build context string from previous cycles for the planner."""
        if not self.cycle_history:
            return ""

        lines = ["Previous cycles:"]
        for cycle in self.cycle_history[-3:]:  # Last 3 cycles
            lines.append(f"\nCycle {cycle['cycle']}:")
            lines.append(f"  Plan rationale: {cycle['plan'].get('rationale', '')}")
            for task in cycle['plan'].get('tasks', []):
                status = task.get('status', 'unknown')
                lines.append(f"  Task {task['id']}: {task['description']} [{status}]")
            test = cycle.get('test_result', {})
            lines.append(f"  Tests: {'PASSED' if test.get('passed') else 'FAILED'} - {test.get('summary', '')}")
            if test.get('failures'):
                for f in test['failures']:
                    lines.append(f"    Failure: {f}")
        return "\n".join(lines)

    def _is_done(self) -> bool:
        if not self.plan:
            return False
        tasks = self.plan.tasks
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)
