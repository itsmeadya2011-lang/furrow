from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
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


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, progress_callback: Callable[[str], Any] | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.progress_callback = progress_callback
        self.state_file = Path(self.client.settings.workspace) / ".furrow_state.json"
        self.history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as e:
                console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _progress(self, message: str) -> None:
        if self.progress_callback:
            try:
                await self.progress_callback(message)
            except Exception:
                pass

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.tasks = plan.tasks
        await self._progress(f"Cycle {self.cycles}: Plan created with {len(plan.tasks)} tasks")
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._progress(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                if isinstance(result, dict) and "files" in result:
                    for file_change in result.get("files", []):
                        try:
                            await self.client.write_file(file_change["path"], file_change["content"])
                            console.print(f"  [dim]Wrote {file_change['path']}[/dim]")
                        except Exception as e:
                            console.print(f"  [red]Failed to write {file_change['path']}: {e}[/red]")
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._progress(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        await self._progress(f"Cycle {self.cycles}: Tests {'passed' if test_result.passed else 'failed'}")

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self.history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": [{"id": t.id, "status": t.status} for t in self.tasks],
            "test_passed": test_result.passed,
            "test_summary": test_result.summary,
        })
        self._save_state()

    def _save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "result": t.result,
                }
                for t in self.tasks
            ],
            "history": self.history,
            "updated_at": datetime.utcnow().isoformat(),
        }
        try:
            self.state_file.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            pass

    @classmethod
    def load_state(cls, state_file: str | Path, client: LLMClient | None = None) -> Orchestrator | None:
        try:
            data = json.loads(Path(state_file).read_text())
            orchestrator = cls(goal=data["goal"], client=client)
            orchestrator.cycles = data.get("cycles", 0)
            orchestrator.tasks = [
                TaskModel(id=t["id"], description=t["description"], status=t.get("status", "pending"), result=t.get("result"))
                for t in data.get("tasks", [])
            ]
            orchestrator.history = data.get("history", [])
            return orchestrator
        except Exception:
            return None

    def _is_done(self) -> bool:
        if not self._get_tasks():
            return True
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
