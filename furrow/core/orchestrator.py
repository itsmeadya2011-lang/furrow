from __future__ import annotations

import asyncio
from typing import Callable, Optional

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
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings if settings is not None else default_settings
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self._latest_cycle_tasks: list[TaskModel] = []

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit({"type": "cycle", "cycle": self.cycles})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                reason = self._halt_reason()
                self._emit({"type": "done", "reason": reason})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._emit({"type": "plan", "cycle": self.cycles, "plan": plan.model_dump()})
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._latest_cycle_tasks = []
            return

        await self._execute_tasks(plan)
        self._latest_cycle_tasks = list(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.last_test_result = test_result
        self._emit(
            {
                "type": "test",
                "cycle": self.cycles,
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": list(test_result.failures),
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, plan: Plan) -> None:
        sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        pending: dict[str, TaskModel] = {t.id: t for t in plan.tasks}
        status: dict[str, str] = {}

        while pending:
            runnable = [
                t for t in pending.values()
                if all(d in status and status[d] == "completed" for d in t.dependencies)
            ]
            if not runnable:
                for t in list(pending.values()):
                    t.status = "failed"
                    t.result = "Dependency failed or circular dependency"
                    self._emit(
                        {
                            "type": "task",
                            "cycle": self.cycles,
                            "task_id": t.id,
                            "status": "failed",
                            "result": t.result,
                        }
                    )
                    console.print(f"[red]Task {t.id} failed: {t.result}[/red]")
                break

            async def run_one(task: TaskModel) -> None:
                async with sem:
                    try:
                        result = await WorkerAgent(task=task, client=self.client).run()
                        task.status = "completed"
                        task.result = result
                    except Exception as e:
                        task.status = "failed"
                        task.result = str(e)
                    status[task.id] = task.status
                    self._emit(
                        {
                            "type": "task",
                            "cycle": self.cycles,
                            "task_id": task.id,
                            "status": task.status,
                            "result": task.result,
                        }
                    )
                    if task.status == "completed":
                        console.print(f"[green]Task {task.id} completed[/green]")
                    else:
                        console.print(f"[red]Task {task.id} failed: {task.result}[/red]")

            await asyncio.gather(*(run_one(t) for t in runnable))
            for t in runnable:
                pending.pop(t.id, None)

        self.tasks.extend(plan.tasks)

    def _halt_reason(self) -> Optional[str]:
        max_cycles = getattr(self.settings, "max_cycles", 0)
        if max_cycles > 0 and self.cycles >= max_cycles:
            return "max cycles reached"
        latest = self._latest_cycle_tasks
        if not latest:
            return "no tasks planned"
        if all(t.status == "completed" for t in latest) and self.last_test_result is not None and self.last_test_result.passed:
            return "goal complete"
        return None

    def _is_done(self) -> bool:
        return self._halt_reason() is not None
