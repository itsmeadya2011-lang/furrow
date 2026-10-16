from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_status: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.on_status = on_status

    def _emit(self, message: str) -> None:
        console.print(message)
        if self.on_status is not None:
            try:
                result = self.on_status(message)
                if asyncio.iscoroutine(result):
                    # Fire-and-forget; results are surfaced via console anyway.
                    asyncio.ensure_future(result)
            except Exception:
                pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if self.settings.max_cycles and self.cycles > self.settings.max_cycles:
                self._emit("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
            if self.on_status is not None:
                await self._emit(f"Plan: {plan.rationale}")
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Merge newly planned tasks into the persistent task list, avoiding duplicates by id.
        known_ids = {t.id for t in self.tasks}
        new_tasks = [t for t in plan.tasks if t.id not in known_ids]
        self.tasks.extend(new_tasks)

        if not new_tasks:
            console.print("[yellow]No new tasks planned. Goal may be complete.[/yellow]")
            return

        max_parallel = max(1, self.settings.max_parallel_tasks)
        results: list[Any] = []
        for i in range(0, len(new_tasks), max_parallel):
            chunk = new_tasks[i : i + max_parallel]
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                chunk_results = await asyncio.gather(
                    *[WorkerAgent(task=task, client=self.client, settings=self.settings).run() for task in chunk],
                    return_exceptions=True,
                )
            results.extend(chunk_results)

        for task, result in zip(new_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if self._get_tasks() and completed >= len(self._get_tasks()):
            return True
        return False
