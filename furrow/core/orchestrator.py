from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()

# Type alias for output callback
OutputCallback = Callable[[str], Awaitable[None]]


@dataclass
class OrchestratorState:
    """Persistent state for the orchestrator across cycles."""
    tasks: dict[str, TaskModel] = field(default_factory=dict)
    cycle_history: list[dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        output_callback: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.config = config or settings
        self.client = client or LLMClient(settings=self.config)
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.state = OrchestratorState()
        self.cycles = 0
        self.output_callback = output_callback

    async def _emit(self, message: str) -> None:
        """Emit a message to both console and callback."""
        console.print(message)
        if self.output_callback:
            await self.output_callback(message)

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.config.max_cycles
        
        while True:
            self.cycles += 1
            
            # Check cycle limit
            if max_cycles > 0 and self.cycles > max_cycles:
                await self._emit(f"[yellow]Max cycles ({max_cycles}) reached. Stopping.[/yellow]")
                break
                
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Provide context from previous cycles to planner
        context = self._build_context()
        
        # Add project context on first cycle
        if self.cycles == 1:
            project_context = await self.planner.get_project_context(str(self.config.workspace))
            if project_context:
                context = project_context + "\n\n" + context
        
        await self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal, context=context)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Update state with new tasks (merging with existing)
        for task in plan.tasks:
            if task.id not in self.state.tasks:
                self.state.tasks[task.id] = task

        await self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.config).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        # Record cycle in history
        self.state.cycle_history.append({
            "cycle": self.cycles,
            "tasks": [t.model_dump() for t in plan.tasks],
            "goal": self.goal,
        })

        await self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client, settings=self.config).run(
            self.goal, list(self.state.tasks.values())
        )

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _build_context(self) -> str:
        """Build context string from previous cycles for the planner."""
        if not self.state.cycle_history:
            return ""
        
        parts = ["## Previous Cycles"]
        for cycle in self.state.cycle_history[-3:]:  # Last 3 cycles
            parts.append(f"\n### Cycle {cycle['cycle']}")
            parts.append(f"Goal: {cycle['goal']}")
            for task in cycle["tasks"]:
                parts.append(f"  - {task['id']}: {task['description']} ({task['status']})")
        
        parts.append("\n## Current Task State")
        for task in self.state.tasks.values():
            parts.append(f"  - {task.id}: {task.description} ({task.status})")
            if task.result:
                parts.append(f"    Result: {task.result[:200]}...")
        
        return "\n".join(parts)

    def _is_done(self) -> bool:
        if not self.state.tasks:
            return False
        
        completed = sum(1 for t in self.state.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.state.tasks.values() if t.status == "failed")
        
        if failed > 0:
            return False
        if completed >= len(self.state.tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return list(self.state.tasks.values())
