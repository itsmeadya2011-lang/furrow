from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import CycleState, Settings, TaskModel
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
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.history: list[CycleState] = []
        self._all_tasks: dict[str, TaskModel] = {}  # task_id -> TaskModel

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
                
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            cycle_state = await self._cycle()
            self.history.append(cycle_state)
            
            if self._is_done(cycle_state):
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> CycleState:
        # Plan
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, self.history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return CycleState(
                cycle_number=self.cycles,
                timestamp=datetime.now(timezone.utc).isoformat(),
                plan=plan,
                tasks=[],
                goal=self.goal,
            )

        # Track tasks
        for task in plan.tasks:
            self._all_tasks[task.id] = task

        # Execute tasks in parallel
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update task statuses
        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        # Test
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return CycleState(
            cycle_number=self.cycles,
            timestamp=datetime.now(timezone.utc).isoformat(),
            plan=plan,
            tasks=plan.tasks,
            test_result=test_result,
            goal=self.goal,
        )

    def _is_done(self, cycle_state: CycleState) -> bool:
        # Check if all tracked tasks are completed
        if not self._all_tasks:
            return False
            
        completed = sum(1 for t in self._all_tasks.values() if t.status == "completed")
        failed = sum(1 for t in self._all_tasks.values() if t.status == "failed")
        
        if failed > 0:
            return False
        return completed >= len(self._all_tasks)

    def get_task_summary(self) -> dict[str, int]:
        """Get summary of all tasks across cycles."""
        status_counts: dict[str, int] = {}
        for task in self._all_tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        return status_counts

    def get_history(self) -> list[CycleState]:
        """Get the full execution history."""
        return self.history

    def print_stats(self) -> None:
        """Print LLM usage statistics."""
        if self.client:
            stats = self.client.get_stats()
            console.print(Panel(
                f"[bold]LLM Usage Statistics[/bold]\n"
                f"Total calls: {stats['total_calls']}\n"
                f"Input tokens: {stats['total_input_tokens']:,}\n"
                f"Output tokens: {stats['total_output_tokens']:,}\n"
                f"Total tokens: {stats['total_tokens']:,}\n"
                f"Estimated cost: ${stats['estimated_cost_usd']:.4f}\n"
                f"Errors: {stats['errors']}",
                title="Statistics",
                border_style="green"
            ))