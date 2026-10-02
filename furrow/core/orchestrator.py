from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger()

OutputCallback = Callable[[str], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []

    async def run(
        self,
        output_callback: OutputCallback | None = None,
    ) -> None:
        self._output_callback = output_callback
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        log.info("orchestrator.start", goal=self.goal, max_cycles=self.settings.max_cycles)
        while True:
            self.cycles += 1
            self._emit(f"═══ Cycle {self.cycles} ═══")
            await self._cycle()
            if self._is_done():
                self._emit("Goal complete. Halting.")
                log.info("orchestrator.done", cycles=self.cycles)
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._emit("Reached max cycles. Halting.")
                log.info("orchestrator.max_cycles_reached", cycles=self.cycles)
                break

    def _emit(self, message: str) -> None:
        console.print(message)
        log.info("orchestrator.output", message=message)
        callback = getattr(self, "_output_callback", None)
        if callback is not None:
            asyncio.ensure_future(callback(message))

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, prev_tasks=self.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("No tasks planned. Goal may be complete.")
            return

        self.tasks = list(plan.tasks)

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[_run_task(task) for task in self.tasks],
                return_exceptions=True,
            )

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            self._emit(f"Tests passed: {test_result.summary}")
            log.info("tester.passed", summary=test_result.summary)
        else:
            self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("Will attempt fix in next cycle.")
            log.warning("tester.failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
