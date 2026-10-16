from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.current_task_ids: set[str] = set()
        self.state_path = Path(settings.workspace) / ".furrow_state.json"

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._load_state()
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Reset statuses so tasks can be re-executed in this cycle.
        for task in self.tasks:
            task.status = "pending"
            task.result = None

        # Persist planned tasks across cycles, avoiding duplicate ids.
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)
                existing_ids.add(task.id)

        self.current_task_ids = {t.id for t in plan.tasks}

        max_parallel = settings.max_parallel_tasks or 0
        if max_parallel > 0:
            sem = asyncio.Semaphore(max_parallel)

            async def run_limited(task: TaskModel) -> str:
                async with sem:
                    return await WorkerAgent(task=task, client=self.client).run()

            runners = [run_limited(task) for task in plan.tasks]
        else:
            runners = [WorkerAgent(task=task, client=self.client).run() for task in plan.tasks]

        with Status("[bold yellow]Executing tasks in parallel...", console=console) as status:
            results = await asyncio.gather(*runners, return_exceptions=True)

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

    def _is_done(self) -> bool:
        if not self.current_task_ids:
            return False
        current = [t for t in self.tasks if t.id in self.current_task_ids]
        failed = sum(1 for t in current if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in current if t.status == "completed")
        return completed >= len(current)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }
        try:
            self.state_path.write_text(json.dumps(state, indent=2, default=str))
        except OSError:
            pass

    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.tasks = []
            return
        self.goal = data.get("goal", self.goal)
        self.cycles = data.get("cycles", 0)
        self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
