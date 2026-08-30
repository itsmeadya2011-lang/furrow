from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.history: list[str] = []

    async def run(self, websocket: Any = None) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if websocket:
            await self._send(websocket, f"Starting Furrow. Goal: {self.goal}")
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            if websocket:
                await self._send(websocket, f"═══ Cycle {self.cycles} ═══")
            if self.client.settings.max_cycles > 0 and self.cycles > self.client.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
                if websocket:
                    await self._send(websocket, f"Reached max_cycles ({self.client.settings.max_cycles}). Halting.")
                break
            await self._cycle(websocket)
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                if websocket:
                    await self._send(websocket, "Goal complete. Halting.")
                break

    async def _cycle(self, websocket: Any = None) -> None:
        context = self._build_context()
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context=context)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if websocket:
            await self._send(websocket, f"Plan: {plan.rationale}\nTasks: {len(plan.tasks)}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            if websocket:
                await self._send(websocket, "No tasks planned. Goal may be complete.")
            self.history.append("Cycle: no tasks planned")
            return

        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                if websocket:
                    await self._send(websocket, f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                if websocket:
                    await self._send(websocket, f"Task {task.id} completed")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            if websocket:
                await self._send(websocket, f"Tests passed: {test_result.summary}")
            self.history.append(f"Cycle {self.cycles}: {len(plan.tasks)} tasks, tests passed")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            if websocket:
                await self._send(websocket, f"Tests failed: {test_result.summary}\n" + "\n".join(f"  • {f}" for f in test_result.failures) + "\nWill attempt fix in next cycle.")
            self.history.append(f"Cycle {self.cycles}: {len(plan.tasks)} tasks, tests failed")

    def _build_context(self) -> str:
        lines = [f"Goal: {self.goal}"]
        if self.history:
            lines.append("\nPrevious cycles:")
            lines.extend(f"- {h}" for h in self.history[-5:])
        completed = [t for t in self.all_tasks if t.status == "completed"]
        if completed:
            lines.append("\nCompleted tasks:")
            lines.extend(f"- {t.id}: {t.description}" for t in completed)
        failed = [t for t in self.all_tasks if t.status == "failed"]
        if failed:
            lines.append("\nFailed tasks:")
            lines.extend(f"- {t.id}: {t.description} ({t.result})" for t in failed)
        return "\n".join(lines)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks

    async def _send(self, websocket: Any, message: str) -> None:
        try:
            await websocket.send_text(message)
        except Exception:
            pass
