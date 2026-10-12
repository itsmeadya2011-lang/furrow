from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.goal = goal
        self.original_goal = goal
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.events: list[dict[str, Any]] = []
        self.on_event: list[Callable[[dict[str, Any]], Any]] = []
        self.state_path = Path(".furrow") / "state.json"

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit({"type": "start", "goal": self.goal})

        while True:
            self.cycles += 1
            self._emit({"type": "cycle_start", "cycle": self.cycles})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()
            await self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit({"type": "done", "cycles": self.cycles})
                break

            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]")
                self._emit({"type": "max_cycles_reached", "cycles": self.cycles})
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit({"type": "plan", "cycle": self.cycles, "plan": plan.model_dump()})

        self.tasks = plan.tasks

        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            await asyncio.gather(*[self._execute_task(t) for t in self.tasks])

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        self.last_test_result = test_result
        self._emit({"type": "test_result", "cycle": self.cycles, "result": test_result.model_dump()})

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_task(self, task: TaskModel) -> None:
        try:
            result_json = await WorkerAgent(task=task, client=self.client).run()
            data = json.loads(result_json)

            for edit in data.get("edits", []):
                path = Path(edit["path"])
                if not path.is_absolute():
                    path = self.settings.workspace / path
                await self.client.write_file(path, edit.get("content", ""))

            task.status = "completed" if data.get("complete", True) else "pending"
            task.result = data.get("summary", "")
            self._emit({"type": "task_complete", "task": task.model_dump()})
            console.print(f"[green]Task {task.id} completed[/green]")
        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            self._emit({"type": "task_failed", "task": task.model_dump(), "error": str(e)})
            console.print(f"[red]Task {task.id} failed: {e}[/red]")

    def _is_done(self) -> bool:
        if not self.tasks:
            return True

        if any(t.status == "failed" for t in self.tasks):
            return False

        if all(t.status == "completed" for t in self.tasks):
            if self.last_test_result is None or self.last_test_result.passed:
                return True

        return False

    async def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "last_test_result": self.last_test_result.model_dump() if self.last_test_result else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.client.write_file(self.state_path, json.dumps(state, indent=2))

    def _emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for callback in self.on_event:
            try:
                result = callback(event)
                if inspect.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                pass