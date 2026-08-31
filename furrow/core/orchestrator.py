from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

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
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int | None = None,
        max_parallel: int | None = None,
        workspace: Path | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.test_passed: Optional[bool] = None
        self.on_event = on_event
        self.max_cycles = max_cycles if max_cycles is not None else self.settings.max_cycles
        self.max_parallel = max_parallel if max_parallel is not None else self.settings.max_parallel_tasks
        self.workspace = Path(workspace) if workspace else self.settings.workspace

    def _emit(self, msg: str) -> None:
        if self.on_event is None:
            return
        try:
            result = self.on_event(msg)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception:
            pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit(f"Furrow started. Goal: {self.goal}")
        while True:
            if self.max_cycles and self.cycles >= self.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                self._emit("Max cycles reached. Halting.")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit(f"Cycle {self.cycles} started.")
            await self._cycle()
            self._persist()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._merge_tasks(plan.tasks)
        console.print(Panel(Pretty([t.model_dump() for t in self.tasks]), title="Tasks", border_style="blue"))
        self._emit(f"Planned {len(self.tasks)} task(s).")

        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            self._emit("No pending tasks this cycle.")
        else:
            sem = asyncio.Semaphore(max(1, self.max_parallel))

            async def run_task(task: TaskModel) -> Any:
                async with sem:
                    worker = WorkerAgent(
                        task=task, client=self.client, workspace=self.workspace
                    )
                    return await worker.run()

            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                results = await asyncio.gather(
                    *(run_task(t) for t in pending), return_exceptions=True
                )

            for task, result in zip(pending, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    self._emit(f"Task {task.id} failed: {result}")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    self._emit(f"Task {task.id} completed.")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        self.test_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._emit(f"Tests failed: {test_result.summary}")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _merge_tasks(self, planned: list[TaskModel]) -> None:
        by_id = {t.id: t for t in self.tasks}
        for new in planned:
            existing = by_id.get(new.id)
            if existing is None:
                self.tasks.append(new)
            elif existing.status == "completed":
                existing.description = new.description
                existing.files = new.files
                existing.dependencies = new.dependencies
            else:
                existing.description = new.description
                existing.files = new.files
                existing.dependencies = new.dependencies
                existing.status = "pending"
                existing.result = None

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if any(t.status == "failed" for t in self.tasks):
            return False
        if self.max_cycles and self.cycles >= self.max_cycles:
            return True
        if all(t.status == "completed" for t in self.tasks):
            if self.test_passed is True:
                return True
        return False

    def _persist(self) -> None:
        try:
            data = {
                "cycles": self.cycles,
                "goal": self.goal,
                "tasks": [t.model_dump() for t in self.tasks],
            }
            path = Path(self.settings.state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
