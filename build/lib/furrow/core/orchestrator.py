from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, TaskModel, settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_path = state_path or (self.settings.workspace / ".furrow_state.json")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                    console.print(f"[yellow]Reached max_cycles={self.settings.max_cycles}. Stopping.[/yellow]")
                    break
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
                await self._save_state()
        finally:
            await self._save_state()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing ones, preserving prior results.
        existing = {t.id: t for t in self.tasks}
        for task in plan.tasks:
            if task.id in existing:
                prev = existing[task.id]
                task.status = prev.status
                task.result = prev.result
            else:
                task.status = "pending"
                task.result = None
        self.tasks = plan.tasks

        fresh = [t for t in self.tasks if t.status != "completed"]
        if not fresh:
            console.print("[green]All planned tasks already completed.[/green]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

            async def _run(task: TaskModel) -> tuple[TaskModel, Any]:
                async with sem:
                    worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                    for attempt in range(max(1, self.settings.max_attempts_per_task)):
                        try:
                            result = await worker.run()
                            return task, result
                        except Exception as e:  # noqa: BLE001
                            last_err = e
                            if attempt < max(1, self.settings.max_attempts_per_task) - 1:
                                task.result = f"Retry {attempt + 1} after: {last_err}"
                            else:
                                raise
                    raise last_err  # type: ignore[name-defined]

            results = await asyncio.gather(*(_run(t) for t in fresh), return_exceptions=True)

        for task, result in zip(fresh, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, self.tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    async def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
            }
            import aiofiles

            async with aiofiles.open(self.state_path, "w") as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:  # noqa: BLE001
            console.print(f"[dim]Could not save state: {e}[/dim]")