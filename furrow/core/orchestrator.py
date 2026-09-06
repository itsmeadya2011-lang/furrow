from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
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


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if 0 < self.client.settings.max_cycles <= self.cycles:
                console.print(
                    f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        previous_context = self._build_previous_context()

        with Status("[bold yellow]Planning...", console=console) as status:
            goal_for_planner = self.goal
            if previous_context:
                goal_for_planner = f"{self.goal}\n\nPrevious context:\n{previous_context}"
            plan = await self.planner.plan(goal_for_planner)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks (respecting dependencies)...", console=console):
            results = await self._execute_with_dependencies(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self._journal(test_result)

    async def _execute_with_dependencies(self, tasks: list[Any]) -> list[Any]:
        task_map = {task.id: task for task in tasks}
        pending = {task.id for task in tasks}
        results: dict[str, Any] = {}

        while pending:
            ready = [
                task
                for task in tasks
                if task.id in pending
                and all(dep in results for dep in task.dependencies)
            ]
            if not ready:
                # Circular dependency or missing dependency; run remaining to surface errors
                ready = [task for task in tasks if task.id in pending]

            batch_tasks = [
                WorkerAgent(task=task, client=self.client).run() for task in ready
            ]
            batch_futures = [asyncio.ensure_future(t) for t in batch_tasks]
            batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)

            for task, result in zip(ready, batch_results):
                results[task.id] = result
                pending.discard(task.id)

        return [results[task.id] for task in tasks]

    def _build_previous_context(self) -> str:
        if not self.history:
            return ""
        parts = []
        for entry in self.history[-3:]:
            parts.append(
                f"- Cycle {entry['cycle']}: goal={entry['goal']}, "
                f"tests={'passed' if entry['test_passed'] else 'failed'}"
            )
            if entry.get("failures"):
                parts.append(f"  Failures: {', '.join(entry['failures'])}")
        return "\n".join(parts)

    def _journal(self, test_result: TestResult) -> None:
        journal_path = self.client.settings.journal_path
        if not journal_path:
            return
        p = Path(journal_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "cycle": self.cycles,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "goal": self.goal,
            "plan": self.plan.model_dump() if self.plan else None,
            "test_passed": test_result.passed,
            "test_summary": test_result.summary,
            "failures": test_result.failures,
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.history.append(entry)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.plan.tasks if self.plan else []
