from __future__ import annotations

import asyncio
import json

import aiofiles
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self, goal: str, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._last_rationale: str | None = None
        self.state_file = self.settings.workspace / ".furrow" / "state.json"
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        self.goal = data.get("goal", self.goal)
        self.cycles = data.get("cycles", 0)
        self._last_rationale = data.get("rationale")
        self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]

    async def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "goal": self.goal,
            "cycles": self.cycles,
            "rationale": getattr(self, "_last_rationale", None),
            "tasks": [t.model_dump() for t in self.tasks],
        }
        async with aiofiles.open(self.state_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(payload, indent=2, default=str))

    def _remove_state(self) -> None:
        try:
            self.state_file.unlink()
        except FileNotFoundError:
            pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._remove_state()
                break
            if (
                self.settings.max_cycles
                and self.settings.max_cycles > 0
                and self.cycles >= self.settings.max_cycles
            ):
                console.print(
                    f"[bold yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/bold yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        self._last_rationale = plan.rationale
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._save_state()
            return

        self.tasks = plan.tasks

        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)
        await self._execute_waves(plan.tasks, sem)

        with Status("[bold yellow]Testing...", console=console):
            test_result: TestResult = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        await self._save_state()

    async def _run_one(self, task: TaskModel, sem: asyncio.Semaphore) -> str:
        async with sem:
            result = await WorkerAgent(task=task, client=self.client).run()
            return result

    async def _execute_waves(self, tasks: list[TaskModel], sem: asyncio.Semaphore) -> None:
        pending = {t.id: t for t in tasks}
        completed_ids: set[str] = set()

        while pending:
            ready = [
                t
                for t in pending.values()
                if all(dep in completed_ids or dep not in pending for dep in t.dependencies)
            ]

            if not ready:
                console.print(
                    "[bold red]Dependency cycle detected; "
                    "running remaining tasks as final batch.[/bold red]"
                )
                ready = list(pending.values())

            results = await asyncio.gather(
                *[self._run_one(t, sem) for t in ready], return_exceptions=True
            )

            for task, result in zip(ready, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                completed_ids.add(task.id)
                del pending[task.id]

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if any(t.status == "failed" for t in self.tasks):
            return False
        return all(t.status == "completed" for t in self.tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
