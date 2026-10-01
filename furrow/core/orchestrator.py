from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)

# A message sink receives plain-text status strings (e.g. for a WebSocket client).
MessageSink: TypeAlias = Callable[[str], Awaitable[None]]


class Orchestrator:
    """Runs the infinite parallel development loop.

    The loop repeats until the goal is complete (all tasks pass tests),
    ``max_cycles`` is reached (when > 0), or the user interrupts.

    Parameters
    ----------
    goal:
        The high-level objective to achieve.
    client:
        Optional ``LLMClient`` override (useful for testing).
    settings:
        Optional ``Settings`` override.
    message_sink:
        Optional async callable that receives plain-text status updates.
        Used by the web server to stream progress over WebSocket.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        message_sink: MessageSink | None = None,
    ) -> None:
        from furrow.config import settings as default_settings

        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.test_results: list[TestResult] = []
        self.last_plan: Plan | None = None
        self.message_sink: MessageSink | None = message_sink

    async def _emit(self, text: str) -> None:
        """Send a status message to the Rich console and an optional sink."""
        console.print(text)
        if self.message_sink is not None:
            try:
                await self.message_sink(text)
            except Exception as exc:  # noqa: BLE001 - intentionally broad
                logger.warning(f"Message sink failed: {exc}")

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        logger.info(
            f"Starting Furrow development loop (goal={self.goal}, max_cycles="
            f"{self.settings.max_cycles})"
        )
        await self._emit(f"--- Furrow started | Goal: {self.goal} ---")
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info(f"Starting cycle {self.cycles}")
            await self._emit(f"Cycle {self.cycles} starting...")

            await self._cycle()

            if self._should_terminate():
                if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                    console.print(
                        f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]"
                    )
                    logger.info(f"Halting: max_cycles reached after {self.cycles} cycles")
                    await self._emit(f"Reached max_cycles ({self.settings.max_cycles}). Halting.")
                else:
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    logger.info(f"Halting: goal complete after {self.cycles} cycles")
                    await self._emit("Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.last_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info(f"Plan created: {len(plan.tasks)} tasks, rationale={plan.rationale}")
        await self._emit(f"Plan created: {len(plan.tasks)} task(s) — {plan.rationale}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.tasks = []
            return

        # Track plan tasks on the instance so _should_terminate can inspect them.
        self.tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error(f"Task {task.id} failed: {result}")
                await self._emit(f"Task {task.id} FAILED: {result}")
            else:
                task.status = "completed"
                task.result = str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info(f"Task {task.id} completed")
                await self._emit(f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, plan.tasks)
        self.test_results.append(test_result)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info(f"Tests passed: {test_result.summary}")
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning(
                f"Tests failed: {test_result.summary}; failures={test_result.failures}"
            )
            for failure in test_result.failures:
                await self._emit(f"TEST FAILURE: {failure}")
            await self._emit("Tests failed. Will attempt fix in next cycle.")
            # Prepend the existing goal so the fix context accumulates.
            self.goal = (
                f"Fix failing tests for goal: {self.goal}\nFailures:\n"
                + "\n".join(test_result.failures)
            )

    def _should_terminate(self) -> bool:
        """Determine whether the development loop should stop.

        Termination conditions (checked in order):
        1. ``max_cycles`` reached (when ``max_cycles > 0``).
        2. All tasks completed AND the latest test suite passed.
        3. If the latest test suite failed, continue to the next cycle.
        4. If there are no tasks (nothing left to plan), stop.
        """
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        # If the most recent test run failed, keep iterating.
        if self.test_results and not self.test_results[-1].passed:
            return False
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        return completed >= len(self._get_tasks())

    def _get_tasks(self) -> list[Any]:
        return self.tasks
