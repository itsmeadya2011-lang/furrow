from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

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
    def __init__(self, goal: str, client: LLMClient | None = None, websocket: Any | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.history: list[dict[str, Any]] = []
        self.websocket = websocket

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._send(f"🚀 Starting Furrow with goal: {self.goal}")
        
        while True:
            # Check max cycles
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                await self._send(f"[yellow]Max cycles ({settings.max_cycles}) reached. Stopping.[/yellow]")
                break
                
            self.cycles += 1
            await self._send(f"\n{'='*50}\n🔄 Cycle {self.cycles}\n{'='*50}")
            await self._cycle()
            if self._is_done():
                await self._send("[bold green]✅ Goal complete. Halting.[/bold green]")
                break

    async def _send(self, message: str) -> None:
        """Send message to console and websocket if available."""
        console.print(message)
        if self.websocket:
            try:
                await self.websocket.send_text(message)
            except Exception:
                pass

    async def _cycle(self) -> None:
        # Build context from history for planner
        context = self._build_context()
        
        await self._send("[yellow]📋 Planning...[/yellow]")
        plan = await self.planner.plan(self.goal, context)
        await self._send(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._send("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._send("[yellow]⚡ Executing tasks in parallel...[/yellow]")
        worker_tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Update tasks with results
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._send(f"[red]❌ Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._send(f"[green]✅ Task {task.id} completed[/green]")
        
        # Add to our task history
        self.tasks.extend(plan.tasks)
        
        # Record cycle in history
        self.history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "results": [
                {"id": t.id, "status": t.status, "result": t.result} 
                for t in plan.tasks
            ]
        })

        await self._send("[yellow]🧪 Testing...[/yellow]")
        test_result = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            await self._send(f"[green]✅ Tests passed: {test_result.summary}[/green]")
        else:
            await self._send(f"[red]❌ Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._send(f"  • {failure}")
            await self._send("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _build_context(self) -> str:
        """Build context string from history for the planner."""
        if not self.history:
            return "No previous cycles."
        
        context_parts = ["Previous cycles:"]
        for h in self.history[-5:]:  # Last 5 cycles
            context_parts.append(f"\nCycle {h['cycle']}:")
            context_parts.append(f"  Goal: {h['goal']}")
            context_parts.append(f"  Plan: {h['plan']['rationale']}")
            for task in h['plan']['tasks']:
                context_parts.append(f"    - {task['id']}: {task['description']} ({task.get('status', 'planned')})")
            for result in h['results']:
                context_parts.append(f"    Result {result['id']}: {result['status']} - {result['result'][:100] if result['result'] else 'N/A'}")
        return "\n".join(context_parts)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False
