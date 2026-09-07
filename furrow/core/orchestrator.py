from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()

StatusCallback = Optional[Callable[[str], Awaitable[None]]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        stop_event: asyncio.Event | None = None,
        on_status: StatusCallback = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.stop_event = stop_event
        self.on_status = on_status
        self.load_state()

    async def _emit(self, message: str) -> None:
        if self.on_status is not None:
            try:
                await self.on_status(message)
            except Exception:
                pass

    @property
    def _state_path(self) -> Path:
        return Path(settings.workspace) / ".furrow" / "state.json"

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit(f"Starting Furrow with goal: {self.goal}")
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                console.print("[bold yellow]Furrow stopped by user.[/bold yellow]")
                await self._emit("Stopped by user.")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(f"Cycle {self.cycles} starting...")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("Goal complete.")
                break

    async def _cycle(self) -> None:
        if not self.current_plan or not self.current_plan.tasks:
            await self._emit("Planning...")
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
            self.current_plan = plan
        else:
            console.print(
                Panel(Pretty(self.current_plan.model_dump()), title="Plan (from state)", border_style="blue")
            )

        if not self.current_plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.save_state()
            return

        await self._emit(f"Executing {len(self.current_plan.tasks)} tasks in parallel...")
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            pending_tasks = [t for t in self.current_plan.tasks if t.status != "completed"]
            if not pending_tasks:
                console.print("[yellow]All tasks already completed from previous session.[/yellow]")
                self.save_state()
                return
            tasks = [
                WorkerAgent(task=task, client=self.client, workspace=settings.workspace).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._emit(f"Task {task.id} completed")

        retried: set[str] = set()
        failed_tasks = [t for t in self.current_plan.tasks if t.status == "failed" and t.id not in retried]
        if failed_tasks:
            await self._emit(f"Retrying {len(failed_tasks)} failed tasks...")
            console.print(f"[yellow]Retrying {len(failed_tasks)} failed tasks...[/yellow]")
            with Status("[bold yellow]Retrying failed tasks...", console=console):
                retry_coroutines = []
                for task in failed_tasks:
                    fix_task = TaskModel(
                        id=task.id,
                        description=f"Fix and retry task:\n{task.description}\n\nPrevious failure: {task.result}",
                        files=task.files,
                        dependencies=task.dependencies,
                    )
                    retry_coroutines.append(WorkerAgent(task=fix_task, client=self.client, workspace=settings.workspace).run())
                retry_results = await asyncio.gather(*retry_coroutines, return_exceptions=True)

            for task, result in zip(failed_tasks, retry_results):
                retried.add(task.id)
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} retry failed: {result}[/red]")
                    await self._emit(f"Task {task.id} retry failed: {result}")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} retry completed[/green]")
                    await self._emit(f"Task {task.id} retry completed")

        await self._emit("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.current_plan.tasks, workspace=settings.workspace)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            await self._emit(f"Tests failed: {test_result.summary}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self.save_state()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.current_plan.tasks if self.current_plan else []

    def save_state(self) -> None:
        state_path = self._state_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.current_plan.tasks] if self.current_plan else [],
        }
        state_path.write_text(json.dumps(state, indent=2))

    def load_state(self) -> None:
        state_path = self._state_path
        if not state_path.exists():
            return
        try:
            data = json.loads(state_path.read_text())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            tasks_data = data.get("tasks", [])
            if tasks_data:
                tasks = [TaskModel(**t) for t in tasks_data]
                self.current_plan = Plan(tasks=tasks, rationale="")
        except Exception:
            pass
