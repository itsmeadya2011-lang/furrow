from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

StateCallback = Callable[[str, dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_path: str | Path | None = None,
        event_callback: StateCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.state_path = Path(state_path) if state_path else Path(".furrow") / "state.json"
        self._event_callback = event_callback

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._load_state()
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle_start", {"cycle": self.cycles, "goal": self.goal})
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("done", {"cycles": self.cycles})
                await self._save_state()
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
                self._emit("max_cycles", {"cycles": self.cycles})
                await self._save_state()
                break
            await self._save_state()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit("plan", plan.model_dump())

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tasks = []
            await self._save_state()
            return

        # Merge new plan tasks into persisted state
        new_tasks = plan.tasks
        existing_by_id = {t.id: t for t in self.tasks}
        merged: list[Any] = []
        for task in new_tasks:
            if task.id in existing_by_id:
                merged.append(existing_by_id[task.id])
            else:
                merged.append(task)
        self.tasks = merged

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.tasks
                if task.status != "completed"
            ]
            pending_tasks = [t for t in self.tasks if t.status != "completed"]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit("task_failed", {"id": task.id, "error": str(result)})
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit("task_completed", {"id": task.id, "result": result})

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit("test_passed", {"summary": test_result.summary})
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = (
                f"Original goal: {self.goal}\n"
                f"Fix failing tests:\n" + "\n".join(test_result.failures)
            )
            self._emit("test_failed", {"summary": test_result.summary, "failures": test_result.failures})

        await self._save_state()

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    async def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            raw_tasks = data.get("tasks", [])
            self.tasks = [
                TaskModel(
                    id=t["id"],
                    description=t["description"],
                    files=t.get("files", []),
                    dependencies=t.get("dependencies", []),
                    status=t.get("status", "pending"),
                    result=t.get("result"),
                )
                for t in raw_tasks
            ]
            console.print(f"[dim]Restored state from {self.state_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    async def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "files": t.files,
                    "dependencies": t.dependencies,
                    "status": t.status,
                    "result": t.result,
                }
                for t in self.tasks
            ],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.state_path)

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._event_callback:
            try:
                self._event_callback(event, payload)
            except Exception:
                pass
