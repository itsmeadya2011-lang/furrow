from __future__ import annotations

import asyncio
import re
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class _LineBuffer:
    """File-like object that captures a Console's rendered output and forwards
    clean text lines to a synchronous callback.

    Rich "live" widgets (e.g. ``Status``) and terminal escape sequences are
    reduced to plain lines so they can be streamed to a client over a
    WebSocket.
    """

    _ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[0-9A-Za-z]")

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        self._flush()
        return len(text)

    def flush(self) -> None:
        self._flush(final=True)

    def isatty(self) -> bool:
        return False

    def _flush(self, final: bool = False) -> None:
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)
        if final and self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""

    def _emit_line(self, line: str) -> None:
        line = self._ANSI_ESCAPE.sub("", line)
        line = line.rstrip("\r")
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        line = line.rstrip()
        if line:
            self._emit(line)


def _make_console(output_callback: Callable[[str], None] | None) -> Console:
    if output_callback is None:
        return Console()
    return Console(
        file=_LineBuffer(output_callback),
        force_terminal=False,
        color_system=None,
        highlight=False,
    )


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        self.output_callback = output_callback
        self.console = _make_console(output_callback)

    async def run(self, output_callback: Callable[[str], None] | None = None) -> None:
        if output_callback is not None:
            self.output_callback = output_callback
            self.console = _make_console(output_callback)
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as exc:
                self.console.print(f"[red]Cycle {self.cycles} crashed: {exc}[/red]")
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        goal = self._planner_goal()
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(goal)
        self.last_plan = plan
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = list(plan.tasks)

        if not self.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        results = await self._execute_tasks()

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await self.tester.run(goal, self.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")

    async def _execute_tasks(self) -> list[str | Exception]:
        concurrency = self.settings.max_parallel_tasks or 5
        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(task: TaskModel) -> str | Exception:
            try:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()
            except Exception as exc:
                self.console.print(f"[red]Worker for task {task.id} raised: {exc}[/red]")
                return exc

        return await asyncio.gather(
            *(_run_one(task) for task in self.tasks),
            return_exceptions=True,
        )

    def _planner_goal(self) -> str:
        goal = self.goal
        if self.last_test_result is not None and not self.last_test_result.passed:
            failures = self.last_test_result.failures
            if failures:
                joined = "\n".join(f"- {failure}" for failure in failures)
                goal = f"{self.goal}\n\nPrevious test failures to address:\n{joined}"
        return goal

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _is_done(self) -> bool:
        tasks = self._get_tasks()

        if not tasks:
            return True

        if any(task.status == "failed" for task in tasks):
            return False

        if any(task.status != "completed" for task in tasks):
            return False

        if self.last_test_result is None:
            return False
        return self.last_test_result.passed
