from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

console = Console()


class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CYCLE_LIMIT = "cycle_limit"


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    cycles: int
    message: str = ""
    test_result: TestResult | None = None

    @property
    def success(self) -> bool:
        return self.status == RunStatus.SUCCESS

    @property
    def exit_code(self) -> int:
        if self.status == RunStatus.SUCCESS:
            return 0
        if self.status == RunStatus.CYCLE_LIMIT:
            return 2
        return 1


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_cycles: int | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.max_cycles = self.settings.max_cycles if max_cycles is None else max_cycles
        if self.max_cycles < 0:
            raise ValueError("max_cycles must be nonnegative")

        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)

        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.last_result: RunResult | None = None

    async def run(self) -> RunResult:
        self.cycles = 0
        self.tasks = []
        self.last_test_result = None
        self.last_result = None

        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while self.max_cycles == 0 or self.cycles < self.max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                test_result = await self._cycle()
            except Exception as exc:
                message = f"Run failed during cycle {self.cycles}: {exc}"
                console.print(f"[red]{message}[/red]")
                result = RunResult(
                    status=RunStatus.FAILURE,
                    cycles=self.cycles,
                    message=message,
                    test_result=self.last_test_result,
                )
                self.last_result = result
                return result

            if self._is_done():
                message = "Goal complete. Halting."
                console.print(f"[bold green]{message}[/bold green]")
                result = RunResult(
                    status=RunStatus.SUCCESS,
                    cycles=self.cycles,
                    message=message,
                    test_result=test_result,
                )
                self.last_result = result
                return result

            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                message = f"Cycle limit of {self.max_cycles} reached."
                console.print(f"[yellow]{message}[/yellow]")
                result = RunResult(
                    status=RunStatus.CYCLE_LIMIT,
                    cycles=self.cycles,
                    message=message,
                    test_result=test_result,
                )
                self.last_result = result
                return result

        message = f"Cycle limit of {self.max_cycles} reached."
        result = RunResult(status=RunStatus.CYCLE_LIMIT, cycles=self.cycles, message=message)
        self.last_result = result
        return result

    async def _cycle(self) -> TestResult | None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = list(plan.tasks)
        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return None

        worker_agents = [WorkerAgent(task=task, client=self.client) for task in self.tasks]
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(*(agent.run() for agent in worker_agents), return_exceptions=True)

        worker_failed = False
        worker_failures: list[str] = []
        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                worker_failed = True
                worker_failures.append(f"Task {task.id} failed: {result}")
                console.print(f"[red]{worker_failures[-1]}[/red]")
            elif task.status == "failed":
                worker_failed = True
                task.result = result
                message = f"Task {task.id} failed: {result}"
                worker_failures.append(message)
                console.print(f"[red]{message}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        try:
            tester = TesterAgent(client=self.client)
            test_result = await tester.run(self.goal, self.tasks)
        except Exception as exc:
            message = f"Tester failed: {exc}"
            console.print(f"[red]{message}[/red]")
            test_result = TestResult(passed=False, summary=str(exc), failures=[str(exc)])
            self.last_test_result = test_result
            raise RuntimeError(message) from exc

        self.last_test_result = test_result
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        if worker_failed:
            raise RuntimeError("Worker task failures: " + "; ".join(worker_failures))

        return test_result

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        return (
            all(task.status == "completed" for task in self.tasks)
            and not any(task.status == "failed" for task in self.tasks)
            and self.last_test_result is not None
            and self.last_test_result.passed
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
