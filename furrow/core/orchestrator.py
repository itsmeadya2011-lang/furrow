from __future__ import annotations

import asyncio
import io
from contextlib import contextmanager
from typing import Any, Protocol

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

console = Console()


class OutputSink(Protocol):
    """Protocol for receiving orchestrator output (e.g., over WebSocket)."""

    async def emit(self, text: str) -> None: ...


class _NoStatus:
    """No-op context manager for Status in non-terminal environments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NoStatus":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _ConsoleSink:
    """Adapts the Rich console to the OutputSink protocol for CLI mode."""

    async def emit(self, text: str) -> None:
        console.print(text)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output: OutputSink | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._last_plan: Plan | None = None
        self._tasks: list[TaskModel] = []
        self._done: bool = False
        self._output: OutputSink = output if output is not None else _ConsoleSink()
        self._streaming = output is not None

    async def run(self) -> None:
        await self._emit_panel(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while not self._done:
            # Respect max_cycles (0 means infinite)
            if (
                self.settings.max_cycles > 0
                and self.cycles >= self.settings.max_cycles
            ):
                await self._emit(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                break

            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                self._done = True
                break

    async def _cycle(self) -> None:
        with self._status("Planning..."):
            plan = await self.planner.plan(self.goal)
        self._last_plan = plan
        self._tasks = list(plan.tasks)
        await self._emit_panel(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks in parallel, respecting max_parallel_tasks limit
        max_parallel = max(1, self.settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_worker(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with self._status(
            f"Executing {len(plan.tasks)} tasks in parallel (max {max_parallel})..."
        ):
            results = await asyncio.gather(
                *[_run_worker(task) for task in plan.tasks], return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with self._status("Testing..."):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks

    def state(self) -> dict[str, Any]:
        """Return serializable state for persistence/inspection."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "done": self._done,
            "tasks": [t.model_dump() for t in self._tasks],
            "last_plan": self._last_plan.model_dump() if self._last_plan else None,
        }

    async def _emit(self, text: str) -> None:
        await self._output.emit(text)

    async def _emit_panel(self, renderable: Any) -> None:
        if self._streaming:
            buf = io.StringIO()
            c = Console(file=buf, color_system=None, force_terminal=False)
            c.print(renderable)
            await self._emit(buf.getvalue())
        else:
            console.print(renderable)

    @contextmanager
    def _status(self, msg: str):
        if self._streaming:
            yield
        else:
            with Status(msg, console=console):
                yield
